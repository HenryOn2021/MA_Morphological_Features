# Classification of Mitral Regurgitation from Cardiac Cine MRI using Clinically-Interpretable Morphological Features

## Pre-requisite
This project is designed to run on Cine long-axis (LAX) views MRIs, reconstructed to 30 phases in DICOM format. 

The label for individual cases are expected to store in a .json file, with a pairing identifier as the patient folder name as the key. Under each patient key, there should have 2 additional key: 'NoVD' (No Valvular Disease) and 'MR' (Mitral Regurgitation), each with a value of 0 or 1.

Example Folder Structure:
```
root_dir/
├──  Patient_1 folder/
│   ├── cine_2ch.dcm
│   ├── cine_3ch.dcm
│   └── cine_4ch.dcm/
├──  Patient_2 folder/
│   ├── cine_2ch.dcm
│   ├── cine_3ch.dcm
│   └── cine_4ch.dcm/
└── Patient_N folder
```

Example Label Structure:
```
{
    'Patient_1': {
        'NoVD': 0,
        'MR': 1
    },
    'Patient_2': {
        'NoVD': 1,
        'MR': 0
    },
    'Patient_N': {
        'NoVD': 0,
        'MR': 1
    },
}
```

## Set-Up
The Python scripts can be to run in Anaconda and its dependencies are listed in the file 'environment.yml':
```sh
# Create environment
conda env create --file=environment.yml
```

## Execution
The script has 2 main functions and each execution is dependent on the previous step: 
- Cine MRI Labeller
- Cine MRI Analysis Pipeline

The first function read in each Cine MRIs in the patient folder for manual labelling of the mitral annulus landmarks from user input.
```sh
# Execute the Cine MRI Labeller
python MA_Morphological_Features_Pipeline_main.py cine_labeller --root_dir path/to/read_in_N_write_out --dcm_dir path/to/images --labels_path path/to/labels.json --timeout int(idel_time_before_exiting) --n int(number_of_phases2skip_labelling) --pathology str(label_only_NoVD_or_MR)
```

The second function perform the proposed method in the literature, i.e. MA morphological feature extraction and MR classification. It can be run as a end-to-end process or individual step based on the index below:
- 0. end-to-end
- 1. Image to World coordinates conversion
- 2. Coordinates interpolation
- 3. Fitting best-fit ellipse and plane and the corresponding feature extraction
- 4. Displacement feature extraction
- 5. Outlier detection
- 6. Minimum Redundancy and Maximum Relevance (MRMR) feature selection
- 7. MR classification

```sh
# Execute the Cine MRI Analysis Pipeline
python main.py cine_pipeline --mode int(index_listed_above) --root_dir path/to/read_in_N_write_out --dcm_dir path/to/images --labels_path path/to/labels.json
```
