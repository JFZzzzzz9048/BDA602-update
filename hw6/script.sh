if ! mariadb -h mariadb -u root -ppassword -e "USE baseball;"

then

mariadb -h mariadb -u root -ppassword -e "CREATE DATABASE baseball;"

mariadb -h mariadb -u root -ppassword baseball < /hw6/baseball.sql

fi

mariadb -h mariadb -u root -ppassword baseball < /hw6/hw2.sql > /hw6/output.txt
