TARGET=$1
PE=$2
java -jar ~/tools/tercom-0.7.25/tercom.7.25.jar -h $PE -r $TARGET -d 0 -o pra -n alignments
