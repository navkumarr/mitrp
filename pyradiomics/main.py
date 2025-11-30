from radiomics import featureextractor
ext = featureextractor.RadiomicsFeatureExtractor("params.yaml")
res = ext.execute("example.nii.gz", "lung_left.nii.gz")