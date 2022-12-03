if ! mariadb -h mariadb -u root -ppassword -e "USE baseball;"

then

    mariadb -h mariadb -u root -ppassword -e "CREATE DATABASE baseball;"

    mariadb -h mariadb -u root -ppassword baseball < baseball.sql

fi

mariadb -h mariadb -u root -ppassword baseball < hw2.sql > /results/output.txt
