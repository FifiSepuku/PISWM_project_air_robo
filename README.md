# PISWM_project_air_robo
PISWM project | YOLO | Python

Estymacja kierunku poruszania się pieszych na podstawie przeszłych klatek.
		# z uwagi na mierną stabilność śledzenia, nie przywiazaliśmy ogromnej wagi do tego programu, bo już czuliśmy, że trzeba zacząć naprawianie od postaw, żeby uzyskać najpierw stabilne wykrycia, a potem sobie komplikować program śledzeniem czy przewidywaniem kierunku,
		# mimo wszystko spróbowaliśmy stworzyć pamięć dla każdego pieszego, w której przechowywane były informacje z kilku przeszłych klatek obrazu - pozycja na obrazie i prędkość,
		# na podstawie tych danych, za pomocą regresji liniowej spróbowaliśmy wyrysować kierunek, w którym porusza się pieszy - dla idealnej scenografi i braku zakłóceń działa,
	# ostatecznie uznaliśmy, że opieranie się na bardzo prymitywnym systemie wykrywania pieszych - samo yolo z progiem wykrywania - nie pozwoli nam na nic więcej, 

Śledzenie pieszych - nadawanie ID.
		# zaczęliśmy od przypisywania ID każdemu pieszemu i sprawdzeniu utrzymania tych ID na kolejnych klatkach - było mierne, jedna postać dostawała w trakcie kilku klatek różne ID, bo czasem yolo ją gubiło, czasem dwóch pieszych zasłoniło kogoś innego itd.
		# zmienialiśmy informacje na podstawie, których śledzeni byli piesi - pozycja na obrazie, odległość i kolor koszulki - im więcej podobnych rzeczy między kolejnymi klatkami obrazu, tym większa pewność, że to osoba o tym samym ID, 
		# osiągnęliśmy w końcu przyzwoity efekt, ludzie zazwyczaj na jednym nagraniu nie zmieniają ID, ale mimo wszystko daleko mu do stabilności,

W pliku config.json zawarto:
• focal_lenth to ogniskowa kamer odczytana z pliku intrinsics.txt dołączone-
go w datasecie,
• baseline to odległość między kamerami stereo,
• confidence_threshold to minimalna pewność detekcji YOLO wymagana do
uznania obiektu za poprawnie wykryty,
• collision_threshold to próg odległości [m] dla ostrzegania o ryzyku kolizji

	
RAPORT
#1 strona tytułowa
#2 wstęp
	#czym są systemy ADAS i po co się je stosuje?
	#koncepcje smartcity - wymienić i odnieść do projektu
	#sztuczne sieci neuronowe - opis, bazowanie
#3 podział prac
	#odpowiedzialność członków zespołu
#4 użyte technologie
	#stack technologiczny - python v3.13, virtualne środowiska, 
	#yolo - opis wersji n, s, m - czym jest, są?
	#SYNTHIA - opis datasetu - jaki? metadane obrazów (tabelka)
#5 badania
	*#przewidywanie kierunku ruchu pieszego
		#regresja liniowa - proste
		#filtr Kalmanna
	#opis kodu
		# RGB 
		# depth+RGB
	#yolo + eksperymenty
		#parametry kamery
		#różnice między wersjami n, s, m - 3 przejazdy dla RGB i kolejne 3 dla RGB+depth
		#wyjaśnić różnice w wykryciu w danej odległości - próg x metrów - obraz + tabela (miejsce pieszego na obrazie w px i odległość)
		#opisać outliery - skrzynki, czy występują?
#6 github
	#czym jest system kontroli wersji?
	#w jaki sposób w nim działaliśmy? - screeny, branche, pullreq, role członków w projekcie, opis commitów, pull i pushy
	
#7 biblio

