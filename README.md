# PISWM_project_air_robo
PISWM project | YOLO | Python

Estymacja kierunku poruszania się pieszych na podstawie przeszłych klatek.
		# z uwagi na mierną stabilność śledzenia, nie przywiazaliśmy ogromnej wagi do tego programu, bo już czuliśmy, że trzeba zacząć naprawianie od postaw, żeby uzyskać najpierw stabilne wykrycia, a potem sobie komplikować program śledzeniem czy przewidywaniem kierunku,
		# mimo wszystko spróbowaliśmy stworzyć pamięć dla każdego pieszego, w której przechowywane były informacje z kilku przeszłych klatek obrazu - pozycja na obrazie i prędkość,
		# na podstawie tych danych, za pomocą regresji liniowej spróbowaliśmy wyrysować kierunek, w którym porusza się pieszy - dla idealnej scenografi i braku zakłóceń działa,
	# ostatecznie uznaliśmy, że opieranie się na bardzo prymitywnym systemie wykrywania pieszych - samo yolo z progiem wykrywania - nie pozwoli nam na nic więcej, 
