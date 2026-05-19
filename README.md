# Interpretable Distillation Reveals that Deep-learning-based Splicing Models Suffer from Pervasive Confounders and Blind Spots

### Simon Liu, Wenjing Zhang, and Oded Regev

## Installation

### Git LFS

This repository uses Git Large File Storage (LFS) to store large datasets.
First, install Git LFS by following [this guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). After installing Git LFS, perform the following steps:

```bash
# Make sure you have installed Git LFS before running the following commands
git lfs install
git clone git:github.com:/regev-lab/splicing-interpretable-distillation.git
cd splicing-interpretable-distillation
git lfs pull
```

### Python Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
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
git clone git@github.com:/Illumina/SpliceAI.git
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

The `distillation/` directory contains distilled model training code and Jupyter notebooks to interpret trained distilled models:

- **`interpret_spliceai_distillation.ipynb`**: Visualization of distilled SpliceAI model.

- **`interpret_pangolin_distillation.ipynb`**: Visualization of distilled Pangolin models.

- **`interpret_alphagenome_distillation.ipynb`**: Visualization of distilled AlphaGenome model.

The `notebooks/` directory contains Jupyter notebooks for various analyses:

- **`Assay_Splicing_Predictions.ipynb`**: Analysis of splicing predictions on experimental splicing assays:
  - Liao et al. 2023: Synthetic sequence splicing assay
  - Baeza-Centurion et al. 2025: FAS exon 6 mutagenesis assay
  - Chong et al. 2019: MFASS assay

- **`CpGMethylation_Analysis.ipynb`**: Analysis of CpG methylation and splicing in Pappalardi et al. 2019 DNMT1-inhibitor dataset.

- **`EnsemblTranscript_CpGComposition.ipynb`**: Analysis of CpG dinucleotide composition (observed/expected ratios) in transcripts.

- **`Genomic_Stop_Codon_Analysis.ipynb`**: Analysis of splicing predictions for genomic exons with and without stop codons.

- **`MAPT_Splicing_Predictions.ipynb`**: Analysis of splicing predictions for MAPT exon 10 variants and their relationship to RNA secondary structure.

- **`Plot_Variant_Scores.ipynb`**: Visualizations of splicing prediction scores for specific variants.

- **`Liao2023_StemLoop_Plots.ipynb`**: Analysis of splicing predictions for stem-loop-containing exons in Liao et al. 2023 dataset.

- **`SpliceAI_CpG_Islands_Predictions.ipynb`**: Visualization of SpliceAI false positive predictions within CpG islands.

- **`SpliceAI_Transcript_Variant_Scorer.ipynb`**: Visualizations of gene sequences scored with SpliceAI.


## Citation

[citation to be added].
