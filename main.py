#Copyright 2018 UNIST under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse
import distutils.util
import os
import tensorflow as tf
import src.models.BEGAN as began


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--flag", type=distutils.util.strtobool, default='0')
    parser.add_argument("-g", "--gpu_number", type=str, default="1")
    parser.add_argument("-p", "--project", type=str, default="MRIGAN_2D_g0.3_d3")

    # Train Data
    parser.add_argument("-d", "--data_dir", type=str, default="./Data/MRI")
    parser.add_argument("-trd", "--dataset", type=str, default="HCP_MRI")
    parser.add_argument("-tro", "--data_opt", type=str, default="crop")
    parser.add_argument("-trs", "--data_size", type=int, default=256)
    parser.add_argument("-ndp", "--num_depth", type=int, default=3)

    # Train Iteration
    parser.add_argument("-n" , "--niter", type=int, default=200)
    parser.add_argument("-ns", "--nsnapshot", type=int, default=5000)
    parser.add_argument("-mx", "--max_to_keep", type=int, default=5)

    # Train Parameter
    parser.add_argument("-b" , "--batch_size", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-m" , "--momentum", type=float, default=0.5)
    parser.add_argument("-m2", "--momentum2", type=float, default=0.999)
    parser.add_argument("-gm", "--gamma", type=float, default=0.3)
    parser.add_argument("-lm", "--lamda", type=float, default=0.001)
    parser.add_argument("-fn", "--filter_number", type=int, default=64)
    parser.add_argument("-z",  "--input_size", type=int, default=256)
    parser.add_argument("-em", "--embedding", type=int, default=256)

    args = parser.parse_args()

    gpu_number = args.gpu_number
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    with tf.device('/gpu:{0}'.format(gpu_number)):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config=config) as sess:
            model = began.BEGAN(args, sess)

            # TRAIN / TEST
            if args.flag:
                model.train(args.flag)
            else:
                model.test(args.flag)

if __name__ == '__main__':
    main()
