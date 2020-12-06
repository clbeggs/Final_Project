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
cyclegan_quickdraw_trees_A='1-C2vb8uooZce_EukwKDRzLvYcKx7ColR'
cyclegan_quickdraw_trees_B='1Un70dWXsz327DZx97ojTeEIPnYftizmV'

if [ -z "$1" ]; then
  echo "Usage: "$0" <model_name>"
  echo "model_names: [pix2pix_dogs,pix2pix_pizza,pix2pix_trees,pix2pix_apples,cyclegan_pizza,cyclegan_trees,cyclegan_apples,cyclegan_quickdraw_trees]"
  exit
fi

model_type=$(echo "${1%%_*}")

if [ $model_type = 'pix2pix' ]; then
  mkdir -p src/checkpoints/$1
  echo "Downloading "$1"..."
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${!1} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="${!1} -O src/checkpoints/$1/latest_net_G.pth && rm -rf /tmp/cookies.txt
elif [ $model_type = 'cyclegan' ]; then
  mkdir -p src/checkpoints/$1
  A=$1'_A'
  B=$1'_B'
  echo "Downloading "$1"..."
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${!A} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="${!A} -O src/checkpoints/$1/latest_net_G_A.pth && rm -rf /tmp/cookies.txt
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${!B} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="${!B} -O src/checkpoints/$1/latest_net_G_B.pth && rm -rf /tmp/cookies.txt
else
  echo $1" is neither a pix2pix nor cyclegan model!"
  exit
fi
