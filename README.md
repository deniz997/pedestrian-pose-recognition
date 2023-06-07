# Information About Data Input

## OpenPose

The JSON input files that are used to train this model will be generated with OpenPose. However, since OpenPose is a very large project, we will **NOT** upload its source to this repository. But the complete system will rely on an installed OpenPose application.

With existing JSON files that are already computed by OpenPose, this model should be able to train on it's own. The only restriction then will be, that no real-time application can be done.

## TCG Dataset

If you want to use the TCG dataset, use this the download: [TCG Dataset](https://drive.google.com/file/d/1N1fr_ngslFSnnzraCCioK929G51QATZR/view) (do not share this link with anyone outside this project). After downloading, put the extracted files of the data and the annotations into `/data/TCG/`. So for example, the data would be located in `/data/TCG/tcg_data.npy`

