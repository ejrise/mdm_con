
for subj in sub-01 sub-02; do
 cd /path/to/${subj}/dwi

 fslmaths dtd_op -sqr temp1
 fslmaths temp1 -sub 1 temp2
 fslmaths temp2 -mul .666 temp3
 fslmaths dtd_ufa -sqrt temp4
 fslmaths temp4 -recip temp5
 fslmaths temp5 -add temp3 temp6
 fslmaths temp6 -sqrt temp7
 fslmaths temp7 -recip temp8
 fslmaths temp8 -mul dtd_op dtd_FA
 
 rm temp*
done 
