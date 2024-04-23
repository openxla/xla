
workspace=$1
SCRIPT_PATH="`dirname \"$0\"`"
bash $SCRIPT_PATH/install_oneapi.sh $workspace remove
rm -rf $workspace
