
#sanal ortamların listelenmesi:
#conda env list

#sanal ortam oluşturma:
# conda create -n *istediğin isim*

#oluşturduğun sanal ortama girme
#conda activate my_env  ----- deactive etmek için --->  conda deactivate


#yüklü paketlerin listelenmesi
#conda list

#paket yükleme
#conda install numpy

#aynı anda birden fazla paket yükleme
# conda install scipy pandas

#paket silme
# conda remove package_name

#belirli bir versiyona göre paket yükleme
# conda install numpy=1.26.3

#paket yükseltme  ve tüm paketleri yükseltme
# conda upgrade numpy
# conda upgrade -all

#pip ile paket yükleme
# pip install pandas
# pip install pandas==2.2.1

#dışarıya aktarma
#conda env export > environment.yaml













