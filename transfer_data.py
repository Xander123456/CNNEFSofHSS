import os
import time
import numpy
import librosa
import argparse

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument("--data_dir_list", type=str, default=[], nargs="*")
    parser.add_argument('--sample_rate', type=int, default=32000)

    args = parser.parse_args()
    print(args)


    if len(args.data_dir_list):
        for p in args.data_dir_list:
            data_dir = os.path.join(os.path.join(args.data_dir, args.dataset), p)
            print(data_dir)

            data_list = sorted([item for item in os.listdir(data_dir) if item.endswith(".wav")])
            print(len(data_list))

            for i in data_list:
                file_path = os.path.join(data_dir, i)

                data, _ = librosa.load(file_path, sr=args.sample_rate)

                save_path = os.path.join(args.save_dir, os.path.splitext(i)[0])
                numpy.save(save_path, data)

            print(f"{data_dir} Done.")

    else:
        data_dir = os.path.join(args.data_dir, args.dataset)
        print(data_dir)

        data_list = sorted([item for item in os.listdir(data_dir) if item.endswith(".wav")])
        print(len(data_list))

        for i in data_list:
            file_path = os.path.join(data_dir, i)

            data, _ = librosa.load(file_path, sr=args.sample_rate)

            save_path = os.path.join(args.save_dir, os.path.splitext(i)[0])
            numpy.save(save_path, data)

        print(f"{data_dir} Done.")


    print("Done!")
    end = time.time()
    print(f"Time: {(end - start):.2f}s")

