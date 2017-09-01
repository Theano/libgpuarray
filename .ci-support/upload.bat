git describe --exact-match HEAD
if errorlevel 1 exit 0
echo "anaconda upload"