import argparse
import os
import yaml
import torch
import numpy as np

from data_loader import get_dataloaders
from diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm

from models import FaceDiff, FaceDiffBeat, FaceDiffDamm
from utils import *


def trainer_diff(args, train_loader, dev_loader, model, diffusion, optimizer, epoch=100, device="cuda"):
    train_losses = []
    val_losses = []

    save_path = os.path.join(args.save_path,args.model)
    os.makedirs(save_path,exist_ok=True)
    with open(os.path.join(save_path,'config.yaml'),'w') as f:
        yaml.dump(args,f)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    iteration = 0

    for e in range(epoch + 1):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)

            # for vocaset reduce the frame rate from 60 to 30
            if args.dataset == 'vocaset':
                vertice = vertice[::2, :]
            vertice = torch.unsqueeze(vertice, 0)

            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot = template.to(device=device), one_hot.to(device=device)

            loss = diffusion.training_losses(
                model,
                x_start=vertice,
                t=t,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                }
            )['loss']

            loss = torch.mean(loss)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                del audio, vertice, template, one_hot
                torch.cuda.empty_cache()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log)))

        train_losses.append(np.mean(loss_log))

        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            # to gpu
            import copy
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)

            # for vocaset reduce the frame rate from 60 to 30
            if args.dataset == 'vocaset':
                vertice = vertice[::2, :]
            vertice = torch.unsqueeze(vertice, 0)

            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

            train_subject = file_name[0].split("_")[0]
            if train_subject in train_subjects_list:        
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]

                loss = diffusion.training_losses(
                    model,
                    x_start=vertice,     
                    t=t,                 
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    }
                )['loss']

                loss = torch.mean(loss)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    one_hot = one_hot_all[:, iter, :]
                    loss = diffusion.training_losses(
                        model,
                        x_start=vertice,     
                        t=t,                    
                        model_kwargs={
                            "cond_embed": audio,
                            "one_hot": one_hot,
                            "template": template,
                        }
                    )['loss']

                    loss = torch.mean(loss)
                    valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)

        val_losses.append(current_loss)
        if e == args.max_epoch or e % 25 == 0 and e != 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'{args.model}_{args.dataset}_{e}.pth'))
            plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))
        print("epcoh: {}, current loss:{:.8f}".format(e + 1, current_loss))

    plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    
    assert torch.cuda.is_available()
    diffusion = create_gaussian_diffusion(args)

    if 'damm' in args.dataset:
        model = FaceDiffDamm(args)
    elif 'beat' in args.dataset:
        model = FaceDiffBeat(
                args,
                vertice_dim=args.vertice_dim,
                latent_dim=args.feature_dim,
                diffusion_steps=args.diff_steps,
                gru_latent_dim=args.gru_dim,
                num_layers=args.gru_layers,
            )
    else:
        model = FaceDiff(
            args,
            vertice_dim=args.vertice_dim,
            latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps,
            gru_latent_dim=args.gru_dim,
            num_layers=args.gru_layers,
        )
    print("model parameters: ", count_parameters(model))
    cuda = torch.device(args.device)

    model = model.to(cuda)
    dataset = get_dataloaders(args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model = trainer_diff(args, dataset["train"], dataset["valid"], model, diffusion, optimizer,
                         epoch=args.max_epoch, device=args.device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='Name of the dataset folder. eg: BIWI')
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--vertice_dim", type=int, default=15069, help='number of vertices - 23370*3 for BIWI dataset')
    parser.add_argument("--feature_dim", type=int, default=256, help='Latent Dimension to encode the inputs to')
    parser.add_argument("--gru_dim", type=int, default=256, help='GRU Vertex decoder hidden size')          ##pas idéal
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=50, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="model_name", help='name of the trained model')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170731_00024_TA FaceTalk_170809_00138_TA")
    parser.add_argument("--input_fps", type=int, default=50,
                        help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=30,
                        help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--diff_steps", type=int, default=1000, help='number of diffusion steps')
    parser.add_argument("--skip_steps", type=int, default=0, help='number of diffusion steps to skip during inference')
    parser.add_argument("--num_samples", type=int, default=1, help='number of samples to generate per audio')
    parser.add_argument("--beta_type", type=str, default="linear",choices=['cosine','linear'],help='Type of beta scheduler')
    parser.add_argument("--template_file", type=str, default="templates.pkl",help='path of the personalized templates')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=get_args()
    main(args)