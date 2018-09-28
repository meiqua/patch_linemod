# patch linemod

## prerequisite

### pysixd

files: params/  pysixd/  t_less_toolkit/  tools/  
copied from [sixd_toolkit](https://github.com/thodan/sixd_toolkit)  
deal with model reading/rendering, datasets reading and evaluation  

### dataset

get dataset under top level folder folder using following cmd  
```
wget -r -np -nH --cut-dirs=1 -R index.html http://ptak.felk.cvut.cz/6DB/public/
```

### library

install opencv3 with contrib rgbd module  
install pybind11  
install open3d(for icp)

pip3 install -r requirements.txt

### steps

in target folder:  
mkdir build  
cd build/  
cmake ..  
make  

in top level folder, if use pybind:  
pip3 install target_folder/  

python3 patch_linemod_test.py