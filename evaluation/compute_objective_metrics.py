import numpy as np
import argparse
import os
import pickle


def main(args):
    train_subject_list = args.train_subjects.split(" ")
    test_subject_list = args.test_subjects.split(" ")
    template_path=os.path.join(args.data_path, args.dataset, args.template_file)
    if args.dataset == "BIWI":
        sentence_list = ["e" + str(i).zfill(2) for i in range(37, 41)]

        with open(template_path, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        with open(os.path.join(args.region_path, "lve.txt")) as f:
            maps = f.read().split(", ")
            mouth_map = [int(i) for i in maps]

        with open(os.path.join(args.region_path, "fdd.txt")) as f:
            maps = f.read().split(", ")
            upper_map = [int(i) for i in maps]
        nr_vertices = 23370

    elif args.dataset == "vocaset":
        sentence_list = [f'sentence{i}' for i in range(21, 41)]

        with open(template_path, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        with open(os.path.join(args.region_path, "lve.txt")) as f:
            maps = f.read().split(",")
            mouth_map = [int(i) for i in maps]

        with open(os.path.join(args.region_path, "fdd.txt")) as f:
            maps = f.read().split(",")
            upper_map = [int(i) for i in maps]
        nr_vertices = 5023
    else:
        nr_vertices = 6172
        sentence_list = [str(i) for i in range(46, 51)]

        with open(template_path, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        with open(os.path.join(args.region_path, "weighted_mouth_mask.txt")) as f:
            maps = [float(line.strip()) for line in f if line]
            mouth_map = []
            for idx, value in enumerate(maps):
                if value > 0.1:
                    mouth_map.append(idx)

        with open(os.path.join(args.region_path, "forehead_mask.txt")) as f:
            maps = [float(line.strip()) for line in f if line]
            upper_map = []
            for idx, value in enumerate(maps):
                if value > 0.4:
                    upper_map.append(idx)

    cnt = 0
    vertices_gt_all = []
    vertices_pred_all = []
    motion_std_difference = []
    abs_motion_std_difference = []

    mve = 0
    num_seq = 0
    for subject in test_subject_list:
        for sentence in sentence_list:
            print(subject, sentence)

            for condition in train_subject_list: 
                if not os.path.exists(os.path.join(args.result_path,args.model, subject + "_" + sentence + "_condition_" + condition + ".npy")):
                    continue
                vertices_gt = np.load(os.path.join(args.gt_path, subject + "_" + sentence + ".npy")).reshape(-1, nr_vertices, 3) 
                vertices_pred = np.load(
                    os.path.join(args.result_path,args.model, subject + "_" + sentence + "_condition_" + condition + ".npy")).reshape(-1,
                                                                                                                    nr_vertices,
                                                                                                                    3)

                vertices_pred = vertices_pred[:vertices_gt.shape[0], :, :]
                vertices_gt = vertices_gt[:vertices_pred.shape[0], :, :]

                print(vertices_pred.shape)
                mve += np.linalg.norm(vertices_gt - vertices_pred, axis = 2).mean(axis=1).mean()


                motion_pred = vertices_pred - templates[subject].reshape(1, nr_vertices, 3)
                motion_gt = vertices_gt - templates[subject].reshape(1, nr_vertices, 3)

                cnt += vertices_gt.shape[0]

                vertices_gt_all.extend(list(vertices_gt))
                vertices_pred_all.extend(list(vertices_pred))

                L2_dis_upper = np.array([np.square(motion_gt[:, v, :]) for v in upper_map])
                L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
                L2_dis_upper = np.sum(L2_dis_upper, axis=2)
                L2_dis_upper = np.std(L2_dis_upper, axis=0)
                gt_motion_std = np.mean(L2_dis_upper)

                L2_dis_upper = np.array([np.square(motion_pred[:, v, :]) for v in upper_map])
                L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
                L2_dis_upper = np.sum(L2_dis_upper, axis=2)
                L2_dis_upper = np.std(L2_dis_upper, axis=0)
                pred_motion_std = np.mean(L2_dis_upper)

                motion_std_difference.append(gt_motion_std - pred_motion_std)
                abs_motion_std_difference.append(np.abs(gt_motion_std - pred_motion_std))
                print(f"{subject}_{sentence}")
                print('FDD: {:.4e}'.format(motion_std_difference[-1]), 'FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))

                num_seq += 1
    print('--------------')
    print('Frame Number: {}'.format(cnt))

    vertices_gt_all = np.array(vertices_gt_all)
    vertices_pred_all = np.array(vertices_pred_all)

    print(vertices_gt_all.shape)


    distances = np.linalg.norm(vertices_gt_all - vertices_pred_all, axis=2)
    mean_distance = np.mean(distances)

    L2_dis_mouth_max = np.array([np.square(vertices_gt_all[:, v, :] - vertices_pred_all[:, v, :]) for v in mouth_map])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)

    print('Mean Vertex Error: {:.4e}'.format(mean_distance))
    print('Lip Vertex Error: {:.4e}'.format(np.mean(L2_dis_mouth_max)))
    print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
    print('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(motion_std_difference)))


def compute_diversity(args):
    train_subject_list = args.train_subjects.split(" ")
    test_subject_list = args.test_subjects.split(" ")


    if args.dataset == "BIWI":
        sentence_list = ["e" + str(i).zfill(2) for i in range(37, 41)]
        nr_vertices = 23370
    elif args.dataset == "vocaset":
        sentence_list = [f'sentence{i}' for i in range(21, 41)]
        nr_vertices = 5023
    else:
        nr_vertices = 6172

        sentence_list = [str(i) for i in range(46, 51)]

    num_seq = 0
    diversity = 0
    for subject in test_subject_list:
        for sentence in sentence_list:

            print(subject, sentence)
            all_pred_seq = []
            for condition in train_subject_list:
                if not os.path.exists(os.path.join(args.result_path,args.model, subject + "_" + sentence + "_condition_" + condition + ".npy")):
                    continue
                vertices_pred = np.load(
                    os.path.join(args.result_path,args.model, subject + "_" + sentence + "_condition_" + condition + ".npy")).reshape(
                    -1,
                    nr_vertices,
                    3)
                all_pred_seq.append(vertices_pred)

            tottal_diff_seq = 0
            n_seq = len(all_pred_seq)
            if n_seq < 2:
                continue
            for i in range(n_seq - 1):
                for j in range(i + 1, n_seq):
                    tottal_diff_seq += np.linalg.norm(all_pred_seq[i] - all_pred_seq[j], axis=2).mean(axis=1).mean()
            tottal_diff_seq /= ((n_seq - 1) * n_seq / 2)
            print(tottal_diff_seq)
            diversity += tottal_diff_seq

            num_seq += 1
    print('-----------')
    print('Diversity: {:.4e}'.format(diversity / num_seq))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170731_00024_TA FaceTalk_170809_00138_TA")
    parser.add_argument("--result_path", type=str, default="result/")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--gt_path", type=str, default="data/vocaset/vertices_npy/")
    parser.add_argument("--region_path", type=str, default="data/vocaset/regions/")
    parser.add_argument("--template_file", type=str, default="templates.pkl")
    parser.add_argument("--model", type=str, default="model_name")
    parser.add_argument("--num_sample", type=str)
    parser.add_argument("--dataset", type=str, default="vocaset")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=get_args()
    main(args)
    compute_diversity(args)
