# Interpretable Distillation Reveals that Deep-learning-based Splicing Models Suffer from Pervasive Confounders and Blind Spots

### Simon Liu, Wenjing Zhang, and Oded Regev

## Installation

### Git LFS

This repository uses Git Large File Storage (LFS) to store large datasets.
First, install Git LFS by following [this guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). After installing Git LFS, perform the following steps:

```bash
# Make sure you have installed Git LFS before running the following commands
git lfs install
git clone https://github.com/regev-lab/splicing-interpretable-distillation.git
cd splicing-interpretable-distillation
git lfs pull
```

Alternatively, if you do not wish to install Git LFS, you can clone the repository using
```bash
git clone https://github.com/regev-lab/splicing-interpretable-distillation.git
cd splicing-interpretable-distillation
```
and then manually download the first two files from [here](https://github.com/regev-lab/splicing-interpretable-distillation/tree/main/data/methylation) into the `data/methylation/` folder and the `distillation_train.csv.gz` file from [here](https://github.com/regev-lab/splicing-interpretable-distillation/tree/main/distillation/data) into the `distillation/data/` folder.

### Python Environment and Dependencies

We recommend creating a virtual Python environment by following these steps:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# To launch Jupyter:
jupyter lab

# Deactivate the environment when done:
deactivate
```

### File Dependencies

Download required reference files and tools into the `dependencies/` directory:

```bash
cd dependencies
# Reference Genome
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/GRCh38.p14.genome.fa.gz
# GENCODE GTF
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.primary_assembly.basic.annotation.gtf.gz
# Ensembl GFF
wget https://ftp.ensembl.org/pub/release-114/gff3/homo_sapiens/Homo_sapiens.GRCh38.114.gff3.gz
# Clone Illumina SpliceAI repository and install
git clone https://github.com/Illumina/SpliceAI.git
cd SpliceAI
python setup.py install
```

## Repository Structure

```
dependencies/         # Reference genomes, annotations, and external tools
data/                 # Experimental assay datasets and splicing predictions
distillation/         # Model distillation code
notebooks/            # Analysis notebooks (see below)
README.md
```

## Analysis Notebooks

The `distillation/` directory contains distilled model training code and Jupyter notebooks to interpret trained distilled models. We recommend running these notebooks on an NVIDIA GPU (estimated run time on a single A100: ~5 minutes vs. on CPU: ~hours).

To run a hyperparameter search for the SpliceAI distilled model (we recommend using a GPU):

```bash
cd distillation
python3 hyperparameter_search.py \
  --experiment_name spliceai_search \
  --num_trials 50 \
  --train_csv /path/to/distillation/data/distillation_train.csv.gz \
  --train_fraction 0.9 \
  --test_csv /path/to/distillation/data/distillation_test.csv.gz \
  --sequence_column random_exon_flanking_25nt \
  --target_column spliceai_avg \
  --data_type sequence \
  --model_architecture conv1d_nbm \
  --loss kl_divergence_logits \
  --num_cpus 8 \
  --num_cpus_per_trial 2 \
  --num_gpus_per_trial 1 \
  --result_dir ~/ray_results
```

Use `--num_gpus_per_trial 0` to run on CPU only. Paths to `--train_csv` and `--test_csv` must be absolute.

- **`interpret_spliceai_distillation.ipynb`**: Visualization of distilled SpliceAI model (Figs. 1B-C, S4).

- **`interpret_pangolin_distillation.ipynb`**: Visualization of distilled Pangolin models (Figs. S2).

- **`interpret_alphagenome_distillation.ipynb`**: Visualization of distilled AlphaGenome model (Fig. S3).

The `notebooks/` directory contains Jupyter notebooks for various analyses:

- **`Assay_Splicing_Predictions.ipynb`**: Analysis of splicing predictions on experimental splicing assays (Figs. 2A, 3A, 4A, S1, S6, S7, S9, S11A).
  - Liao et al. 2023: Synthetic sequence splicing assay
  - Baeza-Centurion et al. 2025: FAS exon 6 mutagenesis assay
  - Chong et al. 2019: MFASS assay

- **`CpGMethylation_Analysis.ipynb`**: Analysis of CpG methylation and splicing in Pappalardi et al. 2019 DNMT1-inhibitor dataset (Fig. S5).

- **`EnsemblTranscript_CpGComposition.ipynb`**: Analysis of CpG dinucleotide composition (observed/expected ratios) in transcripts (Fig. 2D).

- **`Genomic_Stop_Codon_Analysis.ipynb`**: Analysis of splicing predictions for genomic exons with and without stop codons (Figs. 3B, S10).

- **`MAPT_Splicing_Predictions.ipynb`**: Analysis of splicing predictions for MAPT exon 10 variants and their relationship to RNA secondary structure (Figs. 4B, S11B).

- **`Plot_Variant_Scores.ipynb`**: Visualizations of splicing prediction scores for specific variants (Figs. 3C, 4C).

- **`Liao2023_StemLoop_Plots.ipynb`**: Analysis of splicing predictions for stem-loop-containing exons in Liao et al. 2023 dataset (Fig. S12).

- **`SpliceAI_CpG_Islands_Predictions.ipynb`**: Visualization of SpliceAI false positive predictions within CpG islands (Fig. 2B).

- **`SpliceAI_Transcript_Variant_Scorer.ipynb`**: Visualizations of gene sequences scored with SpliceAI (Figs. 2C, S8).


## Citation

[citation to be added].
