from collections import Counter
import numpy as np
import pandas as pd


def GC_content(seq):
    """
    Calculate the GC content of a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence.

    Returns
    -------
    float
        Fraction of G and C bases in the sequence.
    """
    return (seq.count("G") + seq.count("C")) / len(seq)


def count_dinucleotide(seq, di):
    """
    Count occurrences of a specific dinucleotide in a sequence.

    Parameters
    ----------
    seq : str
        DNA sequence.
    di : str
        Two-character dinucleotide to count (e.g., 'CG', 'GC').

    Returns
    -------
    int
        Number of occurrences of the dinucleotide.
    """
    return seq.count(di)


def dinucleotide_obs_exp(seq, di):
    """
    Calculate the observed/expected ratio of a dinucleotide.

    The expected frequency is calculated as the product of the
    individual nucleotide frequencies.

    Parameters
    ----------
    seq : str
        DNA sequence.
    di : str
        Two-character dinucleotide (e.g., 'CG').

    Returns
    -------
    float
        Observed/expected ratio. Values < 1 indicate depletion,
        values > 1 indicate enrichment.
    """
    assert len(di) == 2, "Dinucleotide must be exactly 2 characters"

    obs = count_dinucleotide(seq, di) / (len(seq) - 1)
    exp = (seq.count(di[0]) / len(seq)) * (seq.count(di[1]) / len(seq))

    return obs / exp if exp > 0 else 0


def count_codons_phase(seq, phase, codons):
    """
    Count codons in a specific reading frame.

    Parameters
    ----------
    seq : str
        DNA sequence.
    phase : int
        Reading frame (0, 1, or 2).
    codons : str or list
        Single codon (str) or list of codons to count.

    Returns
    -------
    int
        Number of matching codons in the specified phase.
    """
    if isinstance(codons, str):
        codons = [codons]

    count = 0
    for i in range(phase, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon in codons:
            count += 1
    return count


def count_codons(seq, codons):
    """
    Count codons across all positions (not phase-specific).

    Parameters
    ----------
    seq : str
        DNA sequence.
    codons : str or list
        Single codon (str) or list of codons to count.

    Returns
    -------
    int
        Number of matching codons anywhere in the sequence.
    """
    if isinstance(codons, str):
        codons = [codons]

    count = 0
    for i in range(0, len(seq) - 2):
        codon = seq[i:i+3]
        if codon in codons:
            count += 1
    return count


def classify_stop_codon_phase(row, codon="stop_codon"):
    """
    Classify which reading frames contain a codon.

    Parameters
    ----------
    row : pd.Series
        DataFrame row with columns like '{codon}_phase0_count', etc.
    codon : str, default='stop_codon'
        Codon prefix for column names.

    Returns
    -------
    str
        Comma-separated list of phases (e.g., '0, 1' or 'None').
    """
    phases = [
        row[f"{codon}_phase0_count"] > 0,
        row[f"{codon}_phase1_count"] > 0,
        row[f"{codon}_phase2_count"] > 0
    ]

    phases_class = [str(i) for i, p in enumerate(phases) if p]
    return ", ".join(phases_class) if phases_class else "None"


def add_CpG_features(df, sequence_col='exon'):
    """
    Add GC content and CpG/GpC dinucleotide features to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a column containing DNA sequences.
    sequence_col : str, default='exon'
        Name of the column containing DNA sequences.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - GC_content: Fraction of G+C bases
        - CpG_obs: Count of CpG dinucleotides
        - GpC_obs: Count of GpC dinucleotides
        - CpG_obs_exp: CpG observed/expected ratio
        - GpC_obs_exp: GpC observed/expected ratio
    """
    # Compute all features at once to avoid dataframe fragmentation
    new_cols = pd.DataFrame({
        "GC_content": df[sequence_col].apply(GC_content),
        "CpG_obs": df[sequence_col].apply(lambda x: count_dinucleotide(x, "CG")),
        "GpC_obs": df[sequence_col].apply(lambda x: count_dinucleotide(x, "GC")),
        "CpG_obs_exp": df[sequence_col].apply(lambda x: dinucleotide_obs_exp(x, "CG")),
        "GpC_obs_exp": df[sequence_col].apply(lambda x: dinucleotide_obs_exp(x, "GC"))
    }, index=df.index)

    return pd.concat([df, new_cols], axis=1)


def add_stop_codon_features(df, sequence_col='exon', other_codons=None):
    """
    Add stop codon features to a DataFrame.

    Calculates counts for stop codons (TAA, TAG, TGA) and optional
    comparison codons, both overall and by reading frame phase.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a column containing DNA sequences.
    sequence_col : str, default='exon'
        Name of the column containing DNA sequences.
    other_codons : list of str, optional
        List of codons to count in addition to stop codons.
        If None, generates all 64 possible codons (excluding stop codons).
        If empty list [], no additional codons are counted.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns for:
        - Stop codon counts by type and phase
        - Comparison codon counts (if other_codons specified)
        - Binary flags for exclusive stop codon presence
        - Phase classification strings
    """
    # Generate all 64 codons if not specified
    if other_codons is None:
        nucleotides = ['A', 'C', 'G', 'T']
        other_codons = [f"{n1}{n2}{n3}"
                       for n1 in nucleotides
                       for n2 in nucleotides
                       for n3 in nucleotides
                       if f"{n1}{n2}{n3}" not in ["TAA", "TAG", "TGA"]]  # Exclude stop codons

    # Collect all new columns in a dictionary to avoid fragmentation
    new_cols = {}

    # Stop codons: phase-specific and total counts
    stop_codons = ["TAA", "TAG", "TGA"]
    for stop_codon in stop_codons:
        phase_counts = []
        for phase in range(3):
            col_name = f"{stop_codon}_phase{phase}_count"
            new_cols[col_name] = df[sequence_col].apply(
                lambda x, p=phase, sc=stop_codon: count_codons_phase(x, p, sc)
            )
            phase_counts.append(new_cols[col_name])
        new_cols[f"{stop_codon}_count"] = sum(phase_counts)

    # Other codons for comparison
    for other_codon in other_codons:
        phase_counts = []
        for phase in range(3):
            col_name = f"{other_codon}_phase{phase}_count"
            new_cols[col_name] = df[sequence_col].apply(
                lambda x, p=phase, oc=other_codon: count_codons_phase(x, p, oc)
            )
            phase_counts.append(new_cols[col_name])
        new_cols[f"{other_codon}_count"] = sum(phase_counts)

    # Total stop codon counts across all types
    new_cols["stop_codon_count"] = sum([
        new_cols[f"{stop_codon}_count"] for stop_codon in stop_codons
    ])

    for phase in range(3):
        new_cols[f"stop_codon_phase{phase}_count"] = sum([
            new_cols[f"{stop_codon}_phase{phase}_count"] for stop_codon in stop_codons
        ])

    # Convert to temporary DataFrame for computing binary flags and classifications
    temp_df = pd.DataFrame(new_cols, index=df.index)

    # Binary flags for exclusive stop codon presence
    new_cols["has_only_TAA"] = (
        (temp_df["TAA_count"] >= 1) &
        (temp_df["TGA_count"] == 0) &
        (temp_df["TAG_count"] == 0)
    )

    new_cols["has_only_TAG"] = (
        (temp_df["TAA_count"] == 0) &
        (temp_df["TGA_count"] == 0) &
        (temp_df["TAG_count"] >= 1)
    )

    new_cols["has_only_TGA"] = (
        (temp_df["TAA_count"] == 0) &
        (temp_df["TGA_count"] >= 1) &
        (temp_df["TAG_count"] == 0)
    )

    # Classify which phases contain each codon type
    for stop_codon in stop_codons:
        new_cols[f"{stop_codon}_phase"] = temp_df.apply(
            lambda x, sc=stop_codon: classify_stop_codon_phase(x, sc), axis=1
        )
    for other_codon in other_codons:
        new_cols[f"{other_codon}_phase"] = temp_df.apply(
            lambda x, oc=other_codon: classify_stop_codon_phase(x, oc), axis=1
        )

    new_cols["stop_codon_phase"] = temp_df.apply(
        lambda x: classify_stop_codon_phase(x, "stop_codon"), axis=1
    )

    # Add all columns at once to avoid fragmentation
    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_cols_df], axis=1)
