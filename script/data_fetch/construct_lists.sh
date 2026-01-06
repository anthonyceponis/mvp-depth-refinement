DIODE_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/diode_indoor_depth"
DIODE_TXT_FILEPATH="$DIODE_TXT_DIRPATH/diode_indoor_train_filename_list.txt"
DIODE_DATADIR="$BASE_DATA_DIR/diode"
mkdir -p $DIODE_TXT_DIRPATH
find $DIODE_DATADIR -type f -name "*png" > $DIODE_TXT_FILEPATH
sed -i "s|$DIODE_DATADIR/||g" $DIODE_TXT_FILEPATH

ARKIT_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/arkitscenes_depth"
ARKIT_TXT_FILEPATH="$ARKIT_TXT_DIRPATH/arkitscenes_train_filename_list.txt"
ARKIT_DATADIR="$BASE_DATA_DIR/arkitscenes_processed"
mkdir -p $ARKIT_TXT_DIRPATH
find $ARKIT_DATADIR -type f -name "*png" > $ARKIT_TXT_FILEPATH
sed -i "s|$ARKIT_DATADIR/||g" $ARKIT_TXT_FILEPATH

WAYMO_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/waymo_depth"
WAYMO_TXT_FILEPATH="$WAYMO_TXT_DIRPATH/waymo_train_filename_list.txt"
WAYMO_DATADIR="$BASE_DATA_DIR/waymo_preprocess"
mkdir -p $WAYMO_TXT_DIRPATH
find $WAYMO_DATADIR -type f -name "*png" > $WAYMO_TXT_FILEPATH
sed -i "s|$WAYMO_DATADIR/||g" $WAYMO_TXT_FILEPATH

HYPERSIM_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/hypersim_depth"

for SPLIT in train val test; do
	HYPERSIM_TXT_FILEPATH="$HYPERSIM_TXT_DIRPATH/hypersim_${SPLIT}_filename_list.txt"
	HYPERSIM_DATADIR="$BASE_DATA_DIR/hypersim_processed/$SPLIT"
	mkdir -p $HYPERSIM_TXT_DIRPATH
	tmp1=$(mktemp)
	tmp2=$(mktemp)
	find $HYPERSIM_DATADIR -type f -name "rgb*" | sort > $tmp1
	find $HYPERSIM_DATADIR -type f -name "depth*" | sort > $tmp2
	paste -d ' ' $tmp1 $tmp2 > $HYPERSIM_TXT_FILEPATH
	sed -i "s|$HYPERSIM_DATADIR/||g" $HYPERSIM_TXT_FILEPATH
	rm $tmp1
	rm $tmp2
	python script/data_fetch/remove_invalid_rgb_images.py --txt_file $HYPERSIM_TXT_FILEPATH --datadir $HYPERSIM_DATADIR
done



