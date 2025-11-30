# Interpretable Distillation Reveals that Deep-learning-based Splicing Models Suffer from Pervasive Confounders and Blind Spots

### Simon Liu, Wenjing Zhang, and Oded Regev

## Installation

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
gunzip GRCh38.p14.genome.fa.gz
# GENCODE GTF
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.primary_assembly.basic.annotation.gtf.gz
gunzip gencode.v49.primary_assembly.basic.annotation.gtf.gz
# Ensembl GFF
wget https://ftp.ensembl.org/pub/release-115/gff3/homo_sapiens/Homo_sapiens.GRCh38.115.gff3.gz
gunzip Homo_sapiens.GRCh38.115.gff3.gz
# Clone Illumina SpliceAI repository
git clone git@github.com:/Illumina/SpliceAI.git
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

- **`SpliceAI_CpG_Islands_Predictions.ipynb`**: Visualization of SpliceAI false positive predictions within CpG islands.

- **`EnsemblTranscript_CpGComposition.ipynb`**: Analysis of CpG dinucleotide composition (observed/expected ratios) in transcripts.

- **`CpGMethylation_Analysis.ipynb`**: Analysis of CpG methylation and splicing in Pappalardi et al. 2019 DNMT1-inhibitor dataset.

- **`Genomic_Stop_Codon_Analysis.ipynb`**: Analysis of splicing predictions for genomic exons with and without stop codons.
  
- **`Plot_Variant_Scores.ipynb`**: Visualizations of splicing prediction scores for specific variants.

- **`MAPT_Splicing_Predictions.ipynb`**: Analysis of splicing predictions for MAPT exon 10 variants and their relationship to RNA secondary structure stability.

- **`SpliceAI_Transcript_Variant_Scorer.ipynb`**: Visualizations of gene sequences scored with SpliceAI.


## Citation

[citation to be added].
