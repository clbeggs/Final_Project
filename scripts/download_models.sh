#!/bin/bash

pix2pix_dogs='10kHCpMi0qXOGNSuL7B289L2E9eviXZbi'
pix2pix_trees='1s_r3Bz87-7WbU1WHj_HI6E3m46wOBUZj'
pix2pix_apples='1onxTYYna-Gys4HeLGCBskEmn3USxbLF6'
pix2pix_pizza='1uKmxHpz7ZfCPWZxhNGefJIcbehpw3hvK'

cyclegan_apples_A='1vtxopfEkG6TwhfCBqdQ3tn9oYHsD3Tsl'
cyclegan_apples_B='17GDSSFYhZFDd12ogIeDS1XM8qXlt6CES'
cyclegan_pizza_A='1SE3YfvBWj9T6ffl39i3t_Sb6v3A_Ycy5'
cyclegan_pizza_B='1SENzw6IGV_Gqh_f7zLsljMK6MTL33kcA'
cyclegan_trees_A='1zBYWhdd-Q7Q5IytASKiXOJCGG91PSQuu'
cyclegan_trees_B='1y66OzoPDET9eHozDgDRQuGkFrmyOqtEP'
# Old cyclegan_quickdraw_trees
#cyclegan_quickdraw_trees_A='1-C2vb8uooZce_EukwKDRzLvYcKx7ColR'
#cyclegan_quickdraw_trees_B='1Un70dWXsz327DZx97ojTeEIPnYftizmV'
cyclegan_quickdraw_trees_A='18FLVQpzn7Yqqta2kKd4JSx_RY_D4NVUj'
cyclegan_quickdraw_trees_B='1UcrgpCLDOLl8OlB5YR_s83axIPqmr8Pl'

mkdir -p src/checkpoints/pix2pix_dogs
mkdir -p src/checkpoints/pix2pix_trees
mkdir -p src/checkpoints/pix2pix_apples
mkdir -p src/checkpoints/pix2pix_pizza

mkdir -p src/checkpoints/cyclegan_apples
mkdir -p src/checkpoints/cyclegan_pizza
mkdir -p src/checkpoints/cyclegan_trees
mkdir -p src/checkpoints/cyclegan_quickdraw_trees

echo "Downloading pix2pix_dogs..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$pix2pix_dogs -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$pix2pix_dogs -O src/checkpoints/pix2pix_dogs/latest_net_G.pth && rm -rf /tmp/cookies.txt

echo "Downloading pix2pix_trees..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$pix2pix_trees -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$pix2pix_trees -O src/checkpoints/pix2pix_trees/latest_net_G.pth && rm -rf /tmp/cookies.txt

echo "Downloading pix2pix_apples..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$pix2pix_apples -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$pix2pix_apples -O src/checkpoints/pix2pix_apples/latest_net_G.pth && rm -rf /tmp/cookies.txt

echo "Downloading pix2pix_pizza..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$pix2pix_pizza -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$pix2pix_pizza -O src/checkpoints/pix2pix_pizza/latest_net_G.pth && rm -rf /tmp/cookies.txt
echo "Done with pix2pix!"

echo "Downloading cyclegan_apples"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$cyclegan_apples_A -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$cyclegan_apples_A -O src/checkpoints/cyclegan_apples/latest_net_G_A.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$cyclegan_apples_B -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$cyclegan_apples_B -O src/checkpoints/cyclegan_apples/latest_net_G_B.pth && rm -rf /tmp/cookies.txt

echo "Downloading cyclegan_pizza"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$cyclegan_pizza_A -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$cyclegan_pizza_A -O src/checkpoints/cyclegan_pizza/latest_net_G_A.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$cyclegan_pizza_B -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$cyclegan_pizza_B -O src/checkpoints/cyclegan_pizza/latest_net_G_B.pth && rm -rf /tmp/cookies.txt

echo "Downloading cyclegan_trees"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$cyclegan_trees_A -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$cyclegan_trees_A -O src/checkpoints/cyclegan_trees/latest_net_G_A.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$cyclegan_trees_B -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$cyclegan_trees_B -O src/checkpoints/cyclegan_trees/latest_net_G_B.pth && rm -rf /tmp/cookies.txt


echo "Downloading cyclegan_quickdraw_trees"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$cyclegan_quickdraw_trees_A -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$cyclegan_quickdraw_trees_A -O src/checkpoints/cyclegan_quickdraw_trees/latest_net_G_A.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$cyclegan_quickdraw_trees_B -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$cyclegan_quickdraw_trees_B -O src/checkpoints/cyclegan_quickdraw_trees/latest_net_G_B.pth && rm -rf /tmp/cookies.txt

echo "Done!"
