# PISWM_project_air_robo
PISWM project | YOLO | Python
Śledzenie pieszych - nadawanie ID.
		# zaczęliśmy od przypisywania ID każdemu pieszemu i sprawdzeniu utrzymania tych ID na kolejnych klatkach - było mierne, jedna postać dostawała w trakcie kilku klatek różne ID, bo czasem yolo ją gubiło, czasem dwóch pieszych zasłoniło kogoś innego itd.
		# zmienialiśmy informacje na podstawie, których śledzeni byli piesi - pozycja na obrazie, odległość i kolor koszulki - im więcej podobnych rzeczy między kolejnymi klatkami obrazu, tym większa pewność, że to osoba o tym samym ID, 
		# osiągnęliśmy w końcu przyzwoity efekt, ludzie zazwyczaj na jednym nagraniu nie zmieniają ID, ale mimo wszystko daleko mu do stabilności,
