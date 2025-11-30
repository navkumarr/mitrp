# Radiomics Texture Feature Families

## GLCM — Gray Level Co-occurrence Matrix  
→ Contrast, Correlation, Homogeneity, JointEntropy  

* GLCM's main function is to loo at how pairs of gray levels occuer at a defined spacial relationship, essentially just comparing the distance and angle of certain grays using different measures.  
* Notably, high entropy + high contrast usually inidcates heterogeneous tissue like tumors or fibrosis, while high homogeneity represents more uniform tissue.  

---

## GLRLM — Gray Level Run Length Matrix  
→ LongRunEmphasis, ShortRunEmphasis  

* GLRLM counts consecutive runs of voxels with the same intensity along a direction.  
* SRE means more fine textures, LRE means long uniform streaks. Latter usually indicates tumors or lesions.  

---

## GLSZM — Gray Level Size Zone Matrix  
→ LargeAreaEmphasis, SmallAreaEmphasis, ZoneEntropy  

* GLSZM groups together clusers of connected voxels with the same gray level.  
* SAE means fine grained, LAE means big homogenous regions, and ZE represents randomness.  
* Any one of them being high represents the magnitdue of their category within the scan.  

---

## GLDM — Gray Level Dependence Matrix  
→ DependenceEntropy, DependenceNonUniformity  

* GLDM measures how much voxels "depend" on their neighbors having similar intensity, with high DE meaning there is complexity and LowGrayLevelEmphasis/HighGrayLevelEmphasis representing darker vs brighter voxels.  
* Again represnts coarse vs smooth mainly.  

---

## NGTDM — Neighborhood Gray Tone Difference Matrix  
→ Coarseness, Contrast, Busyness  

* NGTDM compares each voxel to the average of its neighbors and directly measures coarseness, contrast, and busyness.  
* High coarseness means smooth, high busyness means complex/noisy.  

---

## Processing the Data  

* In terms of how to actually process this data, it may make sense to first come up with a robust set of data that is considered across the board, meaning certain statistical categories in all the above families that show indications of certain conditions.  
* Would need a lot of good data for this but it is possible and can be the direction we take with this project.  

To Do:
Built in pyradiomics into TotalSegmentator
Can look at TotalSegmentator Dataset to establish a workflow
Can look at a possible deep learning model to pull out correlations in data
Look at chunk by chunk dataset
