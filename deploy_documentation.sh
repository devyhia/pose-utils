AWS_PROFILE=personal
PACKAGE_NAME=PoseUtils

source venv/bin/activate
pdoc --html $PACKAGE_NAME --force
cp -r imgs html/$PACKAGE_NAME
AWS_PROFILE=$AWS_PROFILE aws s3 cp  --recursive html/$PACKAGE_NAME s3://docs.abouelnaga.io/$PACKAGE_NAME
deactivate
