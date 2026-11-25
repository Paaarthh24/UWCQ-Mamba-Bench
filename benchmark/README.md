# UWCQ-Bench metadata

One folder per dataset with the light metadata of UWCQ-Bench. The images, masks, per-render measurement ground truth and
rendering parameters are in the dataset zips (see the main README, Download).

- `split_author.json`: the original authors' split (source photographs per split).
- `split_author_testsel.json` (UW-CamCrack789, UW-CrackMap, UW-Crack500): the authors' split as used in the paper,
  with checkpoint selection on the test split as in the MixerCSeg protocol.
- `split_group_7_1_2.json` (UW-DeepCrack): the source-disjoint 7:1:2 split used in the paper (selection on validation).
- `measurement_gt.json`, `measurement_gt_sources.csv`: measurement ground truth per source photograph (all 30 renders
  of a photograph share its mask and therefore its values). `*_px_cons` columns are the 8-orientation consensus used in
  the paper.
- `summary.json`: image counts per split and the median crack contrast retained per water type and depth.
