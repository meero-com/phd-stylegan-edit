path_to_aligned="/home/asya/work_git/recherche/StyleGAN/aligned_images/*.png"
for file in $path_to_aligned; do
	python train_latent.py --im_path $file --lr 0.05 --video --nb_it 10000 --use_enc
done