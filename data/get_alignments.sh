cat $1 | grep 'Alignment' | cut -d ' ' -f2- | tr ' ' 'A' | sed 's/[()]//g' | sed 's/\(.\)/\1 /g'
