sleep 10
if ! mariadb -h mariadb -u root -ppassword -e "USE baseball;"

then

    mariadb -h mariadb -u root -ppassword -e "CREATE DATABASE baseball;"

    mariadb -h mariadb -u root -ppassword baseball < baseball.sql
    mariadb -h mariadb -u root -ppassword baseball < final.sql

fi

mariadb -h mariadb -u root -ppassword baseball < final.sql > /results/output.txt


python3 final.py