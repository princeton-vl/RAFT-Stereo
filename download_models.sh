#!/bin/bash
mkdir models -p
cd models
wget "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1" -O models.zip
unzip models.zip
rm models.zip -f
