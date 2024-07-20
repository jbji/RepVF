### compiling waymo cli

Here we provide a step-by-step guide on how to compile the waymo toolkit.

For more details, refer to the [Waymo Open Dataset Quick Start Guide](https://github.com/waymo-research/waymo-open-dataset/blob/r1.3/docs/quick_start.md).

1. Clone the Waymo Open Dataset repository:

   ```shell
   git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
   cd waymo-od
   ```
2. Navigate to the source directory:

   ```shell
   cd src
   ```
3. Run the following command to test all metrics:

   ```shell
   bazel test waymo_open_dataset/metrics:all
   ```
4. Install the necessary dependencies:

   ```shell
   sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
   ```
5. Download and install Bazel (version 5.4.0):

   ```shell
   BAZEL_VERSION=5.4.0
   wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
   sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
   ```
6. Install build essentials:

   ```shell
   sudo apt install build-essential
   ```
7. If you don't have sudo permissions, add a prefix and run Bazel locally:

   ```shell
   bash bazel-5.4.0-installer-linux-x86_64.sh --prefix=path-to-your/waymo-od/bazel-local
   path-to-your/waymo-od/bazel-local/bin/bazel test waymo_open_dataset/metrics:all
   ```
8. To test metrics again, use this command:

   ```shell
   bazel test waymo_open_dataset/metrics:all
   ```
9. Build the compute detection metrics tool:

   ```shell
   bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
   bazel build waymo_open_dataset/metrics/tools/compute_detection_let_metrics_main 
   ```
10. Run the tool with your data:

    ```shell
    bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
    waymo_open_dataset/metrics/tools/fake_predictions.bin \
    waymo_open_dataset/metrics/tools/fake_ground_truths.bin
    ```

Make sure to replace the file paths with your actual data.
