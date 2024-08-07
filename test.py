import argparse
import os

import pandas as pd
import torch
import numpy as np

from data_loader import get_dataloaders

from models import FaceDiff, FaceDiffBeat, FaceDiffDamm
from utils import *

@torch.no_grad()
def test_diff(args, model, test_loader, epoch, diffusion, device="cuda"):
    result_path = os.path.join(args.result_path,args.model)
    os.makedirs(result_path,exist_ok=True)

    save_path = os.path.join(args.save_path,args.model)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, f'{args.model}_{args.dataset}_{epoch}.pth'),map_location='cuda:0'))
    model = model.to(torch.device(device))
    model.eval()

    sr = 16000
    for audio, vertice, template, one_hot_all, file_name in test_loader:  
        vertice = vertice_path = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)
        if args.dataset == 'vocaset':
            vertice = vertice[::2, :]
        vertice = torch.unsqueeze(vertice, 0)


        audio, vertice =  audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

        num_frames = int(audio.shape[-1] / sr * args.output_fps)
        shape = (1, num_frames - 1, args.vertice_dim) if num_frames < vertice.shape[1] else vertice.shape

        train_subject = file_name[0].split("_")[0]
        vertice_path = os.path.split(vertice_path)[-1][:-4]
        print(vertice_path)

        if train_subject in train_subjects_list or args.dataset == 'beat':  
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            one_hot = one_hot.to(device=device)

            for sample_idx in range(1, args.num_samples + 1):
                sample = diffusion.p_sample_loop(
                    model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    },
                    skip_timesteps=args.skip_steps,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                sample = sample.squeeze()
                sample = sample.detach().cpu().numpy()

                if args.dataset == 'beat':
                    out_path = f"{vertice_path}.npy"
                else:
                    if args.num_samples != 1:
                        out_path = f"{vertice_path}_condition_{condition_subject}_{sample_idx}.npy"
                    else:
                        out_path = f"{vertice_path}_condition_{condition_subject}.npy"
                if 'damm' in args.dataset:
                    sample = RIG_SCALER.inverse_transform(sample)
                    np.save(os.path.join(args.result_path,args.model, out_path), sample)
                    df = pd.DataFrame(sample)
                    df.to_csv(os.path.join(args.result_path,args.model, f"{vertice_path}.csv"), header=None, index=None)
                else:
                    np.save(os.path.join(args.result_path,args.model, out_path), sample)

        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                one_hot = one_hot.to(device=device)

                # sample conditioned
                sample_cond = diffusion.p_sample_loop(
                    model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    },
                    skip_timesteps=args.skip_steps,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                prediction_cond = sample_cond.squeeze()
                prediction_cond = prediction_cond.detach().cpu().numpy()

                prediction = prediction_cond
                if 'damm' in args.dataset:
                    prediction = RIG_SCALER.inverse_transform(prediction)
                    df = pd.DataFrame(prediction)
                    df.to_csv(os.path.join(args.result_path,args.model, f"{vertice_path}.csv"), header=None, index=None)
                else:
                    np.save(os.path.join(args.result_path,args.model, f"{vertice_path}_condition_{condition_subject}.npy"), prediction)


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
   
    test_diff(args, model, dataset["test"], args.max_epoch, diffusion, device=args.device)
    print('End')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='Name of the dataset folder. eg: BIWI')
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--vertice_dim", type=int, default=15069, help='number of vertices - 23370*3 for BIWI dataset')
    parser.add_argument("--feature_dim", type=int, default=256, help='Latent Dimension to encode the inputs to')
    parser.add_argument("--gru_dim", type=int, default=256, help='GRU Vertex decoder hidden size')
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=50, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="model_name", help='name of the trained model')
    parser.add_argument("--save_path", type=str, default="save/", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result/", help='path to the predictions')
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