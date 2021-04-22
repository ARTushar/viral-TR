if [ -d beds ]; then
    rm -rf beds
fi

mkdir beds
cd narrowpeaks

for file in *; do
    echo $file
    extension="${file##*.}"
    filename="${file%.*}"
    echo $filename.bed
    echo '-'
    cut -f 1-6 $file > ../beds/$filename.bed
done
