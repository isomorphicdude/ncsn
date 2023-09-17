import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw, ImageFont, ImageOps

__all__ = ["AnnealRunner"]

GRID_SIZE = 8


class AnnealRunner:
    def __init__(self, args, config, extra_args=None):
        self.args = args
        self.config = config
        self.extra_args = extra_args

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == "Adam":
            return optim.Adam(
                parameters,
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                betas=(self.config.optim.beta1, 0.999),
                amsgrad=self.config.optim.amsgrad,
            )
        elif self.config.optim.optimizer == "RMSProp":
            return optim.RMSprop(
                parameters,
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
            )
        elif self.config.optim.optimizer == "SGD":
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError(
                "Optimizer {} not understood.".format(self.config.optim.optimizer)
            )

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose(
                [transforms.Resize(self.config.data.image_size), transforms.ToTensor()]
            )
        else:
            tran_transform = transforms.Compose(
                [
                    transforms.Resize(self.config.data.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
                [transforms.Resize(self.config.data.image_size), transforms.ToTensor()]
            )

        if self.config.data.dataset == "CIFAR10":
            dataset = CIFAR10(
                os.path.join(self.args.run, "datasets", "cifar10"),
                train=True,
                download=True,
                transform=tran_transform,
            )
            test_dataset = CIFAR10(
                os.path.join(self.args.run, "datasets", "cifar10_test"),
                train=False,
                download=True,
                transform=test_transform,
            )
        elif self.config.data.dataset == "MNIST":
            dataset = MNIST(
                os.path.join(self.args.run, "datasets", "mnist"),
                train=True,
                download=True,
                transform=tran_transform,
            )
            test_dataset = MNIST(
                os.path.join(self.args.run, "datasets", "mnist_test"),
                train=False,
                download=True,
                transform=test_transform,
            )

        elif self.config.data.dataset == "CELEBA":
            if self.config.data.random_flip:
                dataset = CelebA(
                    root=os.path.join(self.args.run, "datasets", "celeba"),
                    split="train",
                    transform=transforms.Compose(
                        [
                            transforms.CenterCrop(140),
                            transforms.Resize(self.config.data.image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                )
            else:
                dataset = CelebA(
                    root=os.path.join(self.args.run, "datasets", "celeba"),
                    split="train",
                    transform=transforms.Compose(
                        [
                            transforms.CenterCrop(140),
                            transforms.Resize(self.config.data.image_size),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                )

            test_dataset = CelebA(
                root=os.path.join(self.args.run, "datasets", "celeba_test"),
                split="test",
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(140),
                        transforms.Resize(self.config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        elif self.config.data.dataset == "SVHN":
            dataset = SVHN(
                os.path.join(self.args.run, "datasets", "svhn"),
                split="train",
                download=True,
                transform=tran_transform,
            )
            test_dataset = SVHN(
                os.path.join(self.args.run, "datasets", "svhn_test"),
                split="test",
                download=True,
                transform=test_transform,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

        test_iter = iter(test_loader)
        self.config.input_dim = (
            self.config.data.image_size**2 * self.config.data.channels
        )

        tb_path = os.path.join(self.args.run, "tensorboard", self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, "checkpoint.pth"))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigmas = (
            torch.tensor(
                np.exp(
                    np.linspace(
                        np.log(self.config.model.sigma_begin),
                        np.log(self.config.model.sigma_end),
                        self.config.model.num_classes,
                    )
                )
            )
            .float()
            .to(self.config.device)
        )

        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == "dsm":
                    loss = anneal_dsm_score_estimation(
                        score, X, labels, sigmas, self.config.training.anneal_power
                    )
                elif self.config.training.algo == "ssm":
                    loss = anneal_sliced_score_estimation_vr(
                        score,
                        X,
                        labels,
                        sigmas,
                        n_particles=self.config.training.n_particles,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256.0 * 255.0 + torch.rand_like(test_X) / 256.0
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(
                        0, len(sigmas), (test_X.shape[0],), device=test_X.device
                    )

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(
                            score,
                            test_X,
                            test_labels,
                            sigmas,
                            self.config.training.anneal_power,
                        )

                    tb_logger.add_scalar(
                        "test_dsm_loss", test_dsm_loss, global_step=step
                    )

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(
                        states,
                        os.path.join(self.args.log, "checkpoint_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log, "checkpoint.pth"))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print(
                    "modulus of grad components: mean {}, max {}".format(
                        grad.abs().mean(), grad.abs().max()
                    )
                )

            return images

    def anneal_Langevin_dynamics(
        self, 
        x_mod, 
        scorenet,
        sigmas, 
        n_steps_each=100, 
        step_lr=0.00002,
        annealing=True
    ):
        images = []

        with torch.no_grad():
            if annealing:
                for c, sigma in tqdm.tqdm(
                    enumerate(sigmas),
                    total=len(sigmas),
                    desc="annealed Langevin dynamics sampling",
                ):
                    labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                    labels = labels.long()
                    
                    # the schedule for annealing
                    step_size = step_lr * (sigma / sigmas[-1]) ** 2

                    # sampling n_steps for each sigma/sigmas[-1]
                    for s in range(n_steps_each):
                        # sample images at each step
                        images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                        noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                        grad = scorenet(x_mod, labels)
                        x_mod = x_mod + step_size * grad + noise
            else:
                for c, _ in tqdm.tqdm(
                    enumerate(sigmas),
                    total=len(sigmas),
                    desc="Not annealed Langevin dynamics sampling",
                ):
                    labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                    labels = labels.long()
                    
                    # the schedule for annealing
                    step_size = step_lr

                    # sampling n_steps for each sigma/sigmas[-1]
                    for s in range(n_steps_each):
                        # sample images at each step
                        images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                        noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                        grad = scorenet(x_mod, labels)
                        x_mod = x_mod + step_size * grad + noise

            return images

    # def anneal_adagrad_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002, beta=0.99):
    #     images = []

    #     with torch.no_grad():
    #         # moving average
    #         m = torch.zeros_like(x_mod)

    #         for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Adagrad Langevin sampling'):
    #             labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
    #             labels = labels.long()

    #             # the schedule for annealing
    #             step_size = step_lr * (sigma / sigmas[-1]) ** 2

    #             # sampling n_steps for each sigma/sigmas[-1]
    #             for s in range(n_steps_each):
    #                 # sample images at each step
    #                 images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
    #                 noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
    #                 grad = scorenet(x_mod, labels)

    #                 # moving average update
    #                 m += grad ** 2

    #                 # update with preconditioning
    #                 x_mod = x_mod + step_size * grad / torch.sqrt(m+1e-7) + noise / torch.sqrt(torch.sqrt(m+1e-7))

    #         return images

    def anneal_rms_Langevin_dynamics(
        self,
        x_mod,
        scorenet,
        sigmas,
        n_steps_each=100,
        step_lr=0.00002,
        beta=0.99,
        annealing=True,
        eps=1e-5,
        use_scalar = False,
        
    ):
        images = []

        with torch.no_grad():
            # moving average
            if annealing:
                for c, sigma in tqdm.tqdm(
                    enumerate(sigmas),
                    total=len(sigmas),
                    desc="annealed RMS Langevin sampling",
                ):
                    labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                    labels = labels.long()

                    # the schedule for annealing
                    step_size = step_lr * (sigma / sigmas[-1]) ** 2
                    
                    m = torch.zeros_like(x_mod)
                        
                    # sampling n_steps for each sigma/sigmas[-1]
                    for s in range(n_steps_each):
                        # sample images at each step
                        images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                        noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                        grad = scorenet(x_mod, labels)

                        # moving average update
                        if not use_scalar:
                            m = beta * m + (1 - beta) * (grad**2)
                        else:
                            m = beta * m + (1 - beta) * torch.mean(grad**2)

                        # update with preconditioning
                        x_mod = (
                            x_mod
                            + step_size * grad / (torch.sqrt(m)+eps)
                            + noise / torch.sqrt(torch.sqrt(m) + eps)
                        )
            else:
                m = torch.zeros_like(x_mod)
                for c, _ in tqdm.tqdm(
                    enumerate(sigmas),
                    total=len(sigmas),
                    desc="RMS Langevin sampling without annealing",
                ):
                    labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                    labels = labels.long()
                    
                    # constant stepsize
                    step_size = step_lr
                
                    # sampling n_steps for each sigma/sigmas[-1]
                    for s in range(n_steps_each):
                        # sample images at each step
                        images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                        noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                        grad = scorenet(x_mod, labels)

                        # moving average update
                        if not use_scalar:
                            m = beta * m + (1 - beta) * (grad**2)
                        else:
                            m = beta * m + (1 - beta) * torch.mean(grad**2)

                        # update with preconditioning
                        # x_mod = (
                        #     x_mod
                        #     + step_size * grad / (torch.sqrt(m)+eps)
                        #     + noise / torch.sqrt(torch.sqrt(m) + eps)
                        # )
                        x_mod = (
                            x_mod
                            + step_size * grad / torch.clamp_min(torch.sqrt(m), eps)
                            + noise / torch.clamp_min(torch.sqrt(torch.sqrt(m)), eps)
                        )

            return images

    def anneal_adam_Langevin_dynamics(
        self,
        x_mod,
        scorenet,
        sigmas,
        n_steps_each=100,
        step_lr=0.00002,
        beta1=0.99,
        beta2=0.999,
        mistaken=False, # does not increment counter
        eps=1e-5,
    ):
        images = []

        with torch.no_grad():
            # moving average
            m = torch.zeros_like(x_mod)
            s = torch.zeros_like(x_mod)
            counter = 0

            for c, sigma in tqdm.tqdm(
                enumerate(sigmas),
                total=len(sigmas),
                desc="annealed ADAM Langevin sampling",
            ):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()

                # the schedule for annealing
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                # sampling n_steps for each sigma/sigmas[-1]
                for s in range(n_steps_each):
                    # sample images at each step
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    
                    print(x_mod.shape)
                    grad = scorenet(x_mod, labels)

                    # moving average update
                    m = beta1 * m + (1 - beta1) * (grad)
                    s = beta2 * s + (1 - beta2) * (grad**2)

                    # bais correction
                    m_hat = m / (1 - beta1 ** (counter + 1))
                    s_hat = s / (1 - beta2 ** (counter + 1))

                    # update with preconditioning
                    x_mod = (
                        x_mod
                        + step_size * m_hat / torch.sqrt(s + eps)
                        + noise / torch.sqrt(torch.sqrt(s_hat + eps))
                    )
                    
                if not mistaken:
                    counter += 1
                    
            return images
        
    # defining the Monge metric SGRLD
    def monge_Langevin_dynamics(
        self,
        x_mod,
        scorenet,
        sigmas,
        n_steps_each=100,
        step_lr=0.00002,
        alpha_2=1e-3,
        annealing=True,
        lambd=0.9,
        use_ema_grad=False,
        shape=(GRID_SIZE**2, 1, 28, 28)
    ):
        images = []
        def get_metrics(dim, grad, alpha_2):
            grad_norm_2 = torch.sum(grad**2)
            G_r = torch.eye(dim) - alpha_2 / (1 + alpha_2 * grad_norm_2) * grad @ grad.T
            G_rsqrt = (
                torch.eye(dim)
                + (1 / torch.sqrt(1 + alpha_2 * grad_norm_2) - 1) / grad_norm_2 * grad @ grad.T
            )
            return G_r, G_rsqrt

        with torch.no_grad():
            # moving average of momentum
            m = torch.zeros_like(x_mod)
            dim = x_mod.shape[0]
            
            for c, sigma in tqdm.tqdm(
                enumerate(sigmas),
                total=len(sigmas),
                desc="Monge Langevin sampling",
            ):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()

                # the schedule for annealing
                if annealing:
                    step_size = torch.tensor(step_lr * (sigma / sigmas[-1]) ** 2)
                else:
                    step_size = torch.tensor(step_lr)

                # sampling n_steps for each sigma/sigmas[-1]
                for s in range(n_steps_each):
                    # sample images at each step
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                    # noise = torch.randn_like(x_mod) * torch.sqrt(step_size * 2)
                    
                    
                    x_mod = x_mod.reshape(shape)
                    print(x_mod.shape)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod.flatten()

                    # moving average update
                    if use_ema_grad:
                        m = lambd * m + (1 - lambd) * (grad)
                    else:
                        m = grad
                        
                    G_r, G_rsqrt = get_metrics(dim, grad, alpha_2)
                    precond_grad = G_r @ grad
                    precond_grad_norm = torch.linalg.norm(precond_grad)
                    
                    if precond_grad_norm > 1e3:
                        factor = torch.linalg.norm(grad) / 1e3
                        dx = (
                            grad / factor * step_size
                            + torch.randn_like(x_mod) / torch.sqrt(factor) * torch.sqrt(step_size * 2)
                        )
                        
                    else:
                        dx = precond_grad * step_size + G_rsqrt @ torch.randn_like(x_mod) \
                            * torch.sqrt(step_size * 2)
                    
                    x_mod = x_mod + dx
                    
                    print(x_mod.shape)

            return images
    
    
    

    # def anneal_nag_Langevin_dynamics(
    #     # nestorev moment (Not used)
    #     self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002, mu=0.9
    # ):  # momentum coefficient in [0,1]
    #     images = []

    #     with torch.no_grad():
    #         # momentum accumulation
    #         v = torch.zeros_like(x_mod)

    #         for c, sigma in tqdm.tqdm(
    #             enumerate(sigmas),
    #             total=len(sigmas),
    #             desc="annealed NAG Langevin sampling",
    #         ):
    #             labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
    #             labels = labels.long()

    #             # the schedule for annealing
    #             step_size = step_lr * (sigma / sigmas[-1]) ** 2

    #             # sampling n_steps for each sigma/sigmas[-1]
    #             for s in range(n_steps_each):
    #                 # sample images at each step
    #                 images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
    #                 noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
    #                 grad = scorenet(x_mod, labels)

    #                 # moving average update
    #                 v = mu * v + step_size * grad

    #                 # update with preconditioning
    #                 x_mod = x_mod + v + noise

    #         return images

    #NOTE: Modify this to return samples
    def test(self, save=False, **kwargs):
        """Return all samples."""
        # Load the model
        states = torch.load(
            os.path.join(self.args.log, "checkpoint.pth"),
            map_location=self.config.device,
        )
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        # the number of sigmas corresponds to the number of classes
        sigmas = np.exp(
            np.linspace(
                np.log(self.config.model.sigma_begin),
                np.log(self.config.model.sigma_end),
                self.config.model.num_classes,
            )
        )

        score.eval()
        grid_size = GRID_SIZE

        imgs = []
        # Only modified this MNIST part
        if self.config.data.dataset == "MNIST":
            samples = torch.rand(grid_size**2, 1, 28, 28, device=self.config.device)

            n_steps = (
                self.extra_args.n_steps if self.extra_args.n_steps is not None else 100
            )
            lr = self.extra_args.lr if self.extra_args.lr is not None else 0.00002

            # choosing the sampler
            if self.extra_args.sampler.lower() == "not_ald":
                annealing = self.extra_args.annealing
                all_samples = self.anneal_Langevin_dynamics(
                    samples, score, sigmas, n_steps, lr, annealing,
                )
            elif self.extra_args.sampler.lower() == "rmsald":
                beta = self.extra_args.beta
                annealing = self.extra_args.annealing
                use_scalar = self.extra_args.use_scalar
                all_samples = self.anneal_rms_Langevin_dynamics(
                    samples, score, sigmas, n_steps, lr, beta, annealing,
                    use_scalar=use_scalar
                )
            elif self.extra_args.sampler.lower() == "adamald":
                beta1 = self.extra_args.beta1
                beta2 = self.extra_args.beta2
                naive = self.extra_args.naive
                all_samples = self.anneal_adam_Langevin_dynamics(
                    samples, 
                    score, 
                    sigmas,
                    n_steps,
                    lr,
                    beta1,
                    beta2,
                    naive
                )
                
            elif self.extra_args.sampler.lower() == "mongeald":
                alpha_2 = self.extra_args.alpha_2
                annealing = self.extra_args.annealing
                lambd = self.extra_args.lambd
                use_ema_grad = self.extra_args.use_ema_grad
                samples = samples.flatten()
                all_samples = self.monge_Langevin_dynamics(
                    samples, score, sigmas, n_steps, 
                    lr, alpha_2, 
                    annealing, lambd, 
                    use_ema_grad
                )
                
            elif self.extra_args.sampler.lower() == "ald":
                annealing = self.extra_args.annealing
                all_samples = self.anneal_Langevin_dynamics(
                    samples, score, sigmas, n_steps, lr, annealing,
                )
            # elif self.extra_args.sampler.lower() == "nagald":
            #     mu = self.extra_args.mu
            #     all_samples = self.anneal_nag_Langevin_dynamics(
            #         samples, score, sigmas, n_steps, lr, mu
            #     )
            # elif self.extra_args.sampler.lower() == 'adagradald':
            #     all_samples = self.anneal_adagrad_Langevin_dynamics(samples, score, sigmas, n_steps, lr)
            
        else:
            samples = torch.rand(grid_size**2, 3, 32, 32, device=self.config.device)
            n_steps = (
                self.extra_args.n_steps if self.extra_args.n_steps is not None else 100
            )
            lr = self.extra_args.lr if self.extra_args.lr is not None else 0.00002
            
            if self.extra_args.sampler.lower() == "ald":
                all_samples = self.anneal_Langevin_dynamics(
                    samples, score, sigmas, n_steps, lr, annealing=self.extra_args.annealing,
                )
            elif self.extra_args.sampler.lower() == "rmsald":
                beta = self.extra_args.beta
                annealing = self.extra_args.annealing
                use_scalar = self.extra_args.use_scalar
                all_samples = self.anneal_rms_Langevin_dynamics(
                    samples, score, sigmas, n_steps, lr, beta, annealing,
                    use_scalar=use_scalar
                )

        # imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
        if save:
            for i, sample in enumerate(
                tqdm.tqdm(all_samples, total=len(all_samples), desc="saving images")
            ):
                sample = sample.view(
                    grid_size**2,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                )

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)

                if i % 10 == 0:
                    # save images every 10 steps
                    im = Image.fromarray(
                        image_grid.mul_(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                    imgs.append(im)

                # image_grid.axes_all
                save_image(
                    image_grid,
                    os.path.join(self.args.image_folder, "image_{}.png".format(i)),
                )
                torch.save(
                    sample,
                    os.path.join(self.args.image_folder, "image_raw_{}.pth".format(i)),
                )
                imgs[0].save(
                    "movie.gif", 
                    save_all=True, 
                    append_images=imgs[1:], 
                    duration=1, loop=0
                )
        
        return all_samples


    def anneal_Langevin_dynamics_inpainting(
        self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100, step_lr=0.000008
    ):
        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 3, 32, 32)
        half_refer_image = refer_image[..., :16]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(
                enumerate(sigmas),
                total=len(sigmas),
                desc="annealed Langevin dynamics sampling",
            ):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                corrupted_half_image = (
                    half_refer_image + torch.randn_like(half_refer_image) * sigma
                )
                x_mod[:, :, :, :16] = corrupted_half_image
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :, :, :16] = corrupted_half_image
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def test_inpainting(self):
        states = torch.load(
            os.path.join(self.args.log, "checkpoint.pth"),
            map_location=self.config.device,
        )
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(
            np.linspace(
                np.log(self.config.model.sigma_begin),
                np.log(self.config.model.sigma_end),
                self.config.model.num_classes,
            )
        )
        score.eval()

        imgs = []
        if self.config.data.dataset == "CELEBA":
            dataset = CelebA(
                root=os.path.join(self.args.run, "datasets", "celeba"),
                split="test",
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(140),
                        transforms.Resize(self.config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            refer_image, _ = next(iter(dataloader))

            samples = torch.rand(
                20,
                20,
                3,
                self.config.data.image_size,
                self.config.data.image_size,
                device=self.config.device,
            )

            all_samples = self.anneal_Langevin_dynamics_inpainting(
                samples, refer_image, score, sigmas, 100, 0.00002
            )
            torch.save(
                refer_image, os.path.join(self.args.image_folder, "refer_image.pth")
            )

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(
                    400,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                )

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(
                        image_grid.mul_(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                    imgs.append(im)

                save_image(
                    image_grid,
                    os.path.join(
                        self.args.image_folder, "image_completion_{}.png".format(i)
                    ),
                )
                torch.save(
                    sample,
                    os.path.join(
                        self.args.image_folder, "image_completion_raw_{}.pth".format(i)
                    ),
                )

        else:
            transform = transforms.Compose(
                [transforms.Resize(self.config.data.image_size), transforms.ToTensor()]
            )

            if self.config.data.dataset == "CIFAR10":
                dataset = CIFAR10(
                    os.path.join(self.args.run, "datasets", "cifar10"),
                    train=True,
                    download=True,
                    transform=transform,
                )
            elif self.config.data.dataset == "SVHN":
                dataset = SVHN(
                    os.path.join(self.args.run, "datasets", "svhn"),
                    split="train",
                    download=True,
                    transform=transform,
                )

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            refer_image, _ = next(data_iter)

            torch.save(
                refer_image, os.path.join(self.args.image_folder, "refer_image.pth")
            )
            samples = torch.rand(
                20,
                20,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size,
            ).to(self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(
                samples, refer_image, score, sigmas, 100, 0.00002
            )

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(
                    400,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                )

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(
                        image_grid.mul_(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                    imgs.append(im)

                save_image(
                    image_grid,
                    os.path.join(
                        self.args.image_folder, "image_completion_{}.png".format(i)
                    ),
                )
                torch.save(
                    sample,
                    os.path.join(
                        self.args.image_folder, "image_completion_raw_{}.pth".format(i)
                    ),
                )

        imgs[0].save(
            os.path.join(self.args.image_folder, "movie.gif"),
            save_all=True,
            append_images=imgs[1:],
            duration=1,
            loop=0,
        )
