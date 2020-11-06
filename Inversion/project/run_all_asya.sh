LANG=en_US # so that we don't have commas for decimals

# declare -i skip=0
# for mse_coef in $(seq 0 10 100)
# do

# for vgg_coef in $(seq 0 10 100)
# do
	
# 	for p_coef in $(seq 0.0 1 2)
# 	do 
# 		echo ""
# 		echo "mse_coef: $mse_coef"
# 		echo "vgg_coef: $vgg_coef"
# 		echo "p_coef: $p_coef"
# 		echo ""

# 		if [[ skip -eq 0 ]]
# 		then
# 			echo "skipping"
# 			skip=1
# 			continue
# 		fi

		
# 		python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/jean_01.png --lr 0.05 --video --nb_it 5000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent 
# 		python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 5000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent
# 	done
# done
# done


# p_coef=1
# vgg_coef=0
# mse_coef=0
# pattern='inc'
# nlatent=18
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=1
# vgg_coef=0
# mse_coef=0
# pattern='dec'
# nlatent=18
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=1
# vgg_coef=0
# mse_coef=0
# pattern='zigup'
# nlatent=18
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=1
# vgg_coef=0
# mse_coef=0
# pattern='zigdown'
# nlatent=18
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=1
# vgg_coef=0
# mse_coef=0
# pattern='inc'
# nlatent=1
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=1
# vgg_coef=0
# mse_coef=0
# pattern='dec'
# nlatent=1
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=1
# vgg_coef=0
# mse_coef=0
# pattern='zigup'
# nlatent=1
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=1
# vgg_coef=0
# mse_coef=0
# pattern='zigdown'
# nlatent=1
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=0
# vgg_coef=1
# mse_coef=0
# pattern='norm'
# nlatent=18
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3


# p_coef=0
# vgg_coef=0
# mse_coef=1
# pattern='norm'
# nlatent=18
# echo ""
# echo "mse_coef: $mse_coef"
# echo "vgg_coef: $vgg_coef"
# echo "p_coef: $p_coef"
# echo "nlatent: $nlatent"
# echo "pattern: $pattern"
# echo ""

# python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3












# run 7
p_coef=1
vgg_coef=0
mse_coef=0
pattern='norm'
nlatent=18
use_enc=''
use_rand=''
echo ""
echo "mse_coef: $mse_coef"
echo "vgg_coef: $vgg_coef"
echo "p_coef: $p_coef"
echo "nlatent: $nlatent"
echo "pattern: $pattern"
echo "use_enc: $use_enc"
echo ""

python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3 $use_enc $use_rand



# run 8
p_coef=1
vgg_coef=0
mse_coef=0
pattern='inc'
nlatent=18
use_enc='--use_enc'
echo ""
echo "mse_coef: $mse_coef"
echo "vgg_coef: $vgg_coef"
echo "p_coef: $p_coef"
echo "nlatent: $nlatent"
echo "pattern: $pattern"
echo "use_enc: $use_enc"
echo ""

python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3 $use_enc



# run 9
p_coef=1
vgg_coef=0
mse_coef=0
pattern='dec'
nlatent=18
use_enc='--use_enc'
echo ""
echo "mse_coef: $mse_coef"
echo "vgg_coef: $vgg_coef"
echo "p_coef: $p_coef"
echo "nlatent: $nlatent"
echo "pattern: $pattern"
echo "use_enc: $use_enc"
echo ""

python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3 $use_enc


# run 8
p_coef=1
vgg_coef=0
mse_coef=0
pattern='zigup'
nlatent=18
use_enc='--use_enc'
echo ""
echo "mse_coef: $mse_coef"
echo "vgg_coef: $vgg_coef"
echo "p_coef: $p_coef"
echo "nlatent: $nlatent"
echo "pattern: $pattern"
echo "use_enc: $use_enc"
echo ""

python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3 $use_enc


# run 8
p_coef=1
vgg_coef=0
mse_coef=0
pattern='zigdown'
nlatent=18
use_enc='--use_enc'
echo ""
echo "mse_coef: $mse_coef"
echo "vgg_coef: $vgg_coef"
echo "p_coef: $p_coef"
echo "nlatent: $nlatent"
echo "pattern: $pattern"
echo "use_enc: $use_enc"
echo ""

python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3 $use_enc


# run 8
p_coef=0
vgg_coef=1
mse_coef=0
pattern='norm'
nlatent=18
use_enc='--use_enc'
echo ""
echo "mse_coef: $mse_coef"
echo "vgg_coef: $vgg_coef"
echo "p_coef: $p_coef"
echo "nlatent: $nlatent"
echo "pattern: $pattern"
echo "use_enc: $use_enc"
echo ""

python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3 $use_enc


# run 8
p_coef=0
vgg_coef=0
mse_coef=1
pattern='norm'
nlatent=18
use_enc='--use_enc'
echo ""
echo "mse_coef: $mse_coef"
echo "vgg_coef: $vgg_coef"
echo "p_coef: $p_coef"
echo "nlatent: $nlatent"
echo "pattern: $pattern"
echo "use_enc: $use_enc"
echo ""

python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3 $use_enc


# run 8
p_coef=1
vgg_coef=0
mse_coef=0
pattern='norm'
nlatent=18
use_enc='--use_enc'
echo ""
echo "mse_coef: $mse_coef"
echo "vgg_coef: $vgg_coef"
echo "p_coef: $p_coef"
echo "nlatent: $nlatent"
echo "pattern: $pattern"
echo "use_enc: $use_enc"
echo ""

python train_latent.py --im_path /home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png --lr 0.05 --video --nb_it 1000 --mse_coef $mse_coef --pvgg_coef $vgg_coef --pdisc_coef $p_coef --nlatent $nlatent --pattern $pattern --percept_fact 1.3 $use_enc


