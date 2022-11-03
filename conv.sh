#!/bin/bash

mkdir -p "train_wav"
for dir in "train"/*; do 
  dir="${dir%/}"
  dir="${dir#"train/"}"
  echo "dir = $dir"

  mkdir "train_wav/$dir" -p
  for file in "train"/$dir/*; do
    file="${file#"train/$dir/"}"
    if [[ "$file" == *".mp3" ]]; then
      echo "mp3 = $file"
      mpg123 -w "train_wav/$dir/${file%.*}.wav" "train/$dir/$file"
    fi
  done
done