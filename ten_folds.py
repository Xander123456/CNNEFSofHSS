import os
import csv
import time
import random
import argparse
import numpy as np

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_dir_list", type=str, default=[], nargs="*")
    parser.add_argument("--csv_file_name", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=1234)

    args = parser.parse_args()
    print(args)

    random.seed(args.random_seed)

    if len(args.data_dir_list):
        label_save_dir = os.path.join(args.save_dir, args.dataset)
        for p in args.data_dir_list:
            csv_file_path = os.path.join(os.path.join(os.path.join(args.data_dir, args.dataset), p), args.csv_file_name)

            with open(csv_file_path, encoding="utf-8") as f:
                data = np.loadtxt(f, str, delimiter=",")

                random_order = [x for x in range(len(data))]
                random.shuffle(random_order)

                l = len(data)
                val_list = list()
                for j in range(10):
                    val_list.append(int(l * j / 10))
                val_list.append(l)

                for i in range(10):
                    val = random_order[val_list[i]:val_list[i + 1]]
                    train = random_order[0:val_list[i]] + random_order[val_list[i + 1]:l]

                    train_save_name = f"{label_save_dir}/{i}/{p}_train.csv"
                    val_save_name = f"{label_save_dir}/{i}/{p}_val.csv"
                    all_train_save_name = f"{label_save_dir}/{i}/train.csv"
                    all_val_save_name = f"{label_save_dir}/{i}/val.csv"

                    with open(train_save_name, "w") as trainf:
                        trainfw = csv.writer(trainf)
                        for m in train:
                            trainfw.writerow(data[m])

                    with open(val_save_name, "w") as valf:
                        valfw = csv.writer(valf)
                        for n in val:
                            valfw.writerow(data[n])

                    with open(all_train_save_name, "a") as all_trainf:
                        all_trainfw = csv.writer(all_trainf)
                        for x in train:
                            all_trainfw.writerow(data[x])

                    with open(all_val_save_name, "a") as all_valf:
                        all_valfw = csv.writer(all_valf)
                        for y in val:
                            all_valfw.writerow(data[y])

            print(f"{csv_file_path} Done.")

    else:
        label_save_dir = os.path.join(args.save_dir, args.dataset)
        csv_file_path = os.path.join(os.path.join(args.data_dir, args.dataset), args.csv_file_name)

        with open(csv_file_path, encoding="utf-8") as f:
            data = np.loadtxt(f, str, delimiter=",")

            random_order = [x for x in range(len(data))]
            random.shuffle(random_order)

            l = len(data)
            val_list = list()
            for i in range(10):
                val_list.append(int(l * i / 10))
            val_list.append(l)

            for i in range(10):
                val = random_order[val_list[i]:val_list[i + 1]]
                train = random_order[0:val_list[i]] + random_order[val_list[i + 1]:l]

                train_save_name = f"{label_save_dir}/{i}/train.csv"
                val_save_name = f"{label_save_dir}/{i}/val.csv"

                with open(train_save_name, "w") as trainf:
                    trainfw = csv.writer(trainf)
                    for i in train:
                        trainfw.writerow(data[i])

                with open(val_save_name, "w") as valf:
                    valfw = csv.writer(valf)
                    for j in val:
                        valfw.writerow(data[j])

            print(f"{csv_file_path} Done.")

    print("Done!")
    end = time.time()
    print(f"Time: {(end - start):.2f}s")
