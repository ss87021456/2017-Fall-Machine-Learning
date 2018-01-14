a=$(pip2 install --user numpy pandas)
cmd_output=$(python2 hw2.py $1 $2)
echo "$cmd_output"
