
function lr_runmydtd(x)

addpath(genpath('/path/to/MatlabCode/md-dmri'))
cd(['/path/to/subfolders/',x,'/sesfolder/dwi/']);

%load in the eight log files corresponding to the eight niftis in same order
subs=dir('log_jmQTI*'); %this information from bval and bvec files, need to be put into log format (see example in qti_preprocess folder) 

[Nn j]=size(subs);
for kk=1:Nn
    mt.diff_array{kk}=dlmread(subs(kk).name,'\t',1,0);
end

%Add a fifth column where linear =1, planar = 0 and spherical = 1/3
for i=1:4
    mt.diff_array{i}(1:end,5)=1;
end

for i=5:8
mt.diff_array{i}(1:end,5)= 0.33333;
end

%combine all the niftis together
snubs=dir('*.nii.gz');
[Np j]=size(snubs);
if Np~=Nn
    error('Error, different numbers of log and nii files')
end

%Need to fix the sort order....%NOTE THIS may not be needed with BIDS
%formatted files, sort order should follow run order of dwi .nii files
mt.nii_img{1} = snubs(4).name;
mt.nii_img{2} = snubs(1).name;
mt.nii_img{3} = snubs(2).name;
mt.nii_img{4} = snubs(3).name;
mt.nii_img{5} = snubs(6).name;
mt.nii_img{6} = snubs(7).name;
mt.nii_img{7} = snubs(8).name;
mt.nii_img{8} = snubs(5).name;

%Convert log files into a format that is compatable with Filips xps
%processing code

jm_setup_xps_and_merge( mt, './' )

clear all

%Eddy and Motion correct using the mio pacakge (requires linux or windows)
setenv('FSLDIR','/sw/fsl');  % this is to tell where FSL folder is
setenv('FSLOUTPUTTYPE', 'NIFTI_GZ')


dir1='/sw/fsl/bin/fslmaths';
if ~exist(dir1,'file')cd 
    fprintf('FOLDER NOT FOUND:%s/n', dir1)
    return
end 

opt=mdm_opt;
s.nii_fn = fullfile(pwd,'MERGED_NII.nii.gz');
load MERGED_NII_xps.mat
s.xps = xps;
o_path = pwd;
s_ref = mdm_s_subsample(s,xps.b <= 1e9, o_path, opt);

p = elastix_p_affine;
p_fn =  elastix_p_write(p, 'p.l');
s_ref = mdm_mec_b0(s_ref,p_fn, o_path,opt);
s_registered = mdm_mec_eb(s,s_ref,p_fn,o_path, opt);

%Make a few fsl calls to cook up a mask

cmd = [dir1, ' MERGED_NII_mc.nii.gz -Tmean mean'];
system(cmd);
cmd = [dir1, ' mean.nii.gz -thr 100 -bin mask'];
system(cmd)
	
%clean up to save some space
!rm MERGED_NII.nii.gz MERGED_NII_sub_mc.nii.gz MERGED_NII_sub.nii.gz MERGED_NII_sub_mc_xps.mat MERGED_NII_tp.txt  MERGED_NII.nii.gz MERGED_NII_sub_ref.nii.gz MERGED_NII_sub_tp.txt MERGED_NII_xps.mat MERGED_NII_ref.nii.gz MERGED_NII_sub_xps.mat p.l

mdm_fit --data 'MERGED_NII_mc.nii.gz' ...
--mask 'mask.nii.gz' ...
--method dtd ...
--out ./. ...
--xps 'MERGED_NII_mc_xps.mat'


