#!/usr/bin/python
# -*- coding: Utf-8 -*

#Parfait pour les images binaires, portées droites

#----------------------------------------------------------
# Importation des librairies

import numpy as np
from pymorph import *
import mahotas as mh
from matplotlib import pylab as plt
import matplotlib.pyplot as plt2
import scipy.ndimage as sc
from math import *
import cv2


#----------------------------------------------------------
# Importation de l'image

img1 = cv2.imread('images/partition2.jpg',0)

# si problème avec la fonction qui grise :  as_grey=True (ne garantit pas des entiers)

#----------------------------------------------------------
# Variables globales empiriques

#longueur max d'un intervalle de noir représentant une portée
seuil_portee = 2
#sert au seuillage
seuil_noir = 200 
#marge autorisée pour les distances entre les lignes de la portée
delta_portee = 2 
#marge autorisée entre les ordonnées des points d'une même ligne de portée
delta_dist = 5
#Nombre de pixels minimum qu'il doit y avoir entre deux barres verticales
entre_bv = 3
#pourcentage de remplissage du carré pour qu'on ait une note
pc_note = 15
#pourcentage de remplissage du carré pour qu'on ait une croche
pc_cro = 45
#pourcentage de remplissage du carré pour qu'on ait une blanche
pc_blan = 30
#Nombre maximal de croches sur une même note
nbr_croches = 1


#----------------------------------------------------------
# Fonctions

#on transforme l'image en 0 et en 1
def img_0_1(img):
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if img[x][y] > seuil_noir:
				img[x][y] = 1
			else:
				img[x][y] = 0

#cherche à détecter les morceaux noirs
def trouve_noir(img,col):
	memoire = 1
	l = []
	for x in range(img.shape[0]):
		if img[x][col] != memoire:
			#plt.plot(col,x-1,'ro')
			memoire = not memoire
			l.append(x-1)
	return l

#parcourt toute la matrice pour trouver les intervalles
def trouve_noir_matrice(img):
	l2 = []
	for x in range(img.shape[1]):
		l = trouve_noir(img,x)
		l2.append(l)
	return l2

#pour une liste donnée, regarde la distance entre deux valeurs consécutives, si elle est assez petite, c'est potentiellement une portée et on en prend le milieu sinon (valeurs isolées) on l'élimine
def les_milieux(l,l2):
	if (len(l) >= 2):
		if l[1]-l[0] > seuil_portee:
			les_milieux(l[1:],l2)
		else:
			l2.append((l[1]+l[0])/2)
			les_milieux(l[2:],l2)
	return l2

#pour toutes les listes d'intervalles noirs, on cherche les milieux et on élimine les points isolés
def les_milieux_liste_de_liste(liste):
	l = []
	for elt in liste:
		l.append(les_milieux(elt,[]))
	return l

def test_cinq(a,b,c,d,e):
	if (b-a) < delta_portee+(c-b) and (c-b) < delta_portee+(d-c) and (d-c) < delta_portee+(e-d) and (e-d) < delta_portee+(b-a):
		rep = 1
	else:
		rep = 0
	return rep

#les listes doivent représenter les portées donc elles doivent être composées de 5/10/15/... points successifs
# problème du bruit ??
def groupe_cinq_points(l,l2):
	if (len(l) >= 5):
		if test_cinq(l[0],l[1],l[2],l[3],l[4]) == 0:
			groupe_cinq_points(l[1:],l2)
		else:
			l2.append([l[0],l[1],l[2],l[3],l[4]])
			groupe_cinq_points(l[5:],l2)
	return l2

#Pour toutes les listes d'intervalles, on ne garde que les points de portée		
def groupe_cinq_points_liste_de_liste(liste):
	l = []
	for elt in liste:
		l.append(groupe_cinq_points(elt,[]))
	return l

#ajoute l'abscisse au bout de chaque liste
def ajoute_abscisse(liste):
	for i in range(len(liste)):
		if len(liste[i]) > 0:
			for elt in liste[i]:
				elt.append(i)
	return liste

#On retire les listes vides (elles sont inutiles)	
def liste_sans_liste_vide(liste):
	l = []
	for elt in liste:
		if len(elt) > 0:
			l.append(elt)
	return l

#on passe de liste de listes de listes à liste de listes
def split_listes(liste):
	l = []
	for elt in liste:
		for elt2 in elt:
			l.append(elt2)
	return l

#donne la première coordonnée de chaque portée (et l'abscisse)
def premiere_coordonnee_liste_de_liste(liste):
	l = []
	for elt in liste:
		l.append((elt[0],elt[5]))
	return l

#donne la deuxième coordonnée de chaque portée (et l'abscisse)
def deuxieme_coordonnee_liste_de_liste(liste):
	l = []
	for elt in liste:
		l.append((elt[1],elt[5]))
	return l

#donne la troisième coordonnée de chaque portée (et l'abscisse)
def troisieme_coordonnee_liste_de_liste(liste):
	l = []
	for elt in liste:
		l.append((elt[2],elt[5]))
	return l

#donne la quatrième coordonnée de chaque portée (et l'abscisse)
def quatrieme_coordonnee_liste_de_liste(liste):
	l = []
	for elt in liste:
		l.append((elt[3],elt[5]))
	return l

#donne la cinquième coordonnée de chaque portée (et l'abscisse)
def cinquieme_coordonnee_liste_de_liste(liste):
	l = []
	for elt in liste:
		l.append((elt[4],elt[5]))
	return l
	
#à partir d'une liste de coordonnées, on sépare les différentes portées (les unes en dessous des autres sur la partition)
def separe_les_portees(liste,l2):
	liste.sort()
	l = []
	r = 1
	if len(liste) > 0:
		for i in range(len(liste)-1):
			if (liste[i][0] > liste[i+1][0]+delta_dist) or (liste[i][0] < liste[i+1][0]-delta_dist):
				l2.append(liste[:i+1])
				liste = liste[i+1:]
				separe_les_portees(liste,l2)
				r = 0 #on ne finit pas la boucle
				break
			else:
				l.append(liste[i])
		if r != 0: #si on a fini la boucle
			l.append(liste[len(liste)-1])
			l2.append(l)
	return l2

def somme_ab_carre(liste):
	som = 0
	for elt in liste:
		som = elt[1]*elt[1] + som
	return som
	
def somme_ab(liste):
	som = 0
	for elt in liste:
		som = elt[1] + som
	return som
	
def somme_ab_ord(liste):
	som = 0
	for elt in liste:
		som = elt[0]*elt[1] + som
	return som
	
def somme_ord(liste):
	som = 0
	for elt in liste:
		som = elt[0] + som
	return som
	
def somme_ord_carre(liste):
	som = 0
	for elt in liste:
		som = elt[0]*elt[0] + som
	return som

#Somme des abscisses au carré = A
#Nombre de points = B
#Somme des abscisses = C
#Somme des abscisses fois ordonnées = D
#Somme des ordonnées = E
#Somme des ordonnées au carré= F
	
def calcul_abcdef(liste):
	a = somme_ab_carre(liste)
	b = len(liste)
	c = somme_ab(liste)
	d = somme_ab_ord(liste)
	e = somme_ord(liste)
	f = somme_ord_carre(liste)
	return (a,b,c,d,e,f)

def calcul_abcdef_plusieurs_listes(liste):
	l = []
	for elt in liste:
		l.append(calcul_abcdef(elt))
	return l

def delta(tup):
	d = tup[0]*tup[1] - tup[2]*tup[2]
	return d

def delta_m(tup):
	d = tup[3]*tup[1] - tup[4]*tup[2]
	return d

def delta_p(tup):
	d = tup[0]*tup[4] - tup[3]*tup[2]
	return d
	
def solution(tup):
	b = 0
	c = 0
	if delta(tup) != 0:
		b = float(float(delta_m(tup))/float(delta(tup)))
		c = float(float(delta_p(tup))/float(delta(tup)))
	return (b,c)
	
def solution_liste(liste):
	l = []
	for elt in liste:
		l.append(solution(elt))
	return l

def tracer_droite(soluce,img):
	x = (0,img.shape[1])
	y = (soluce[1],soluce[1]+soluce[0]*img.shape[1])
	plt.plot(x,y,color = 'blue')
	

def tracer_droite_liste(liste,img):
	for elt in liste:
			tracer_droite(elt,img)

#liste de listes de couples (pente,ordonnée à l'origine)
def moyenne_pentes(liste):
	somme = 0
	compt = 0
	for elt in liste:
		for elt2 in elt:
			somme = elt2[0] + somme
			compt = 1 + compt
	return somme/compt

def ecart_moyen(liste):
	somme = 0
	compt = 0
	for i in range(len(liste)-1):
		for j in range(len(liste[i])):
			somme = abs(liste[i][j][1]-liste[i+1][j][1]) + somme
			compt = 1 + compt
	return somme/compt



#effectue le changement de repère sur l'image
def changement_repere(img,pente):
	img2 = np.zeros(img.shape,np.int)
	if pente != 0:
		teta = -atan(pente) #le repère est "inversé"
		(x0,y0) = (img.shape[1]/2,img.shape[0]/2) #rotation depuis le milieu de l'image
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				try:
					img2[i][j] = img[round((i-x0)*cos(teta)-(j-y0)*sin(teta)+x0)][round((i-x0)*sin(teta)+(j-y0)*cos(teta)+y0)]
				except: 0
	return img2

def remet_listes_cinq(liste):
	i=0
	l=[]
	while i < len(liste):
		l.append(liste[i:i+5])
		i=i+5
	return l
	
#donne les nouvelles valeurs des ordonnées à l'origine après changement de repère
def changement_de_repere_tableau(img,liste,pente):
	teta = -atan(pente)
	(x0,y0) = (img.shape[1]/2,img.shape[0]/2)
	u = []
	l = []
	for elt in liste:
		for elt2 in elt:
			a = round(-x0*sin(teta)+(elt2[1]-y0)*cos(teta)+y0)
			u.append(a)
	return remet_listes_cinq(u)

#trace les droites de pente zéro
def tracer_droite_hori(soluce,img):
	x = (0,img.shape[1])
	y = (soluce,soluce)
	plt.plot(x,y,color = 'blue')

def tracer_droite_hori_liste(liste,img):
	for elt in liste:
		for elt2 in elt:
			tracer_droite_hori(elt2,img)

#fait la soustraction deux à deux de chaque pixel des deux images
def soustraction_img(img1,img2):
	if (img1.shape[0] == img2.shape[0]) and (img1.shape[1] == img2.shape[1]):
		img = np.zeros(img1.shape,np.int)
		for x in range(img1.shape[0]):
			for y in range(img2.shape[1]):
				if (img1[x][y] == img2[x][y]) and (img1[x][y] == 1):
					img[x][y] = 1
				elif (img1[x][y] == img2[x][y]) and (img1[x][y] == 0):
					img[x][y] = 1
				elif (img1[x][y] != img2[x][y]) and (img1[x][y] == 1):
					img[x][y] = 1
				elif (img1[x][y] != img2[x][y]) and (img1[x][y] == 0):
					img[x][y] = 0
	return img



#trouve les barres verticales
def trouve_vertical(img,col):
	memoire = 1
	l = []
	for x in range(img.shape[0]):
		if img[x][col] != memoire:
			#plt.plot(col,x,'ro')
			memoire = not memoire
			l.append(x)
	return l	

def trouve_barres_verticales(img):
	l2 = []
	for x in range(img.shape[1]):
		l = trouve_vertical(img,x)
		l2.append(l)
	return l2
	
	
	
def garde_longues_barres(l,l2,taille_bv):
	if (len(l) >= 2):
		if l[1]-l[0] > taille_bv:
			l2.extend([l[0],l[1]])
			garde_longues_barres(l[2:],l2,taille_bv)
		else:
			garde_longues_barres(l[2:],l2,taille_bv) #peut planter, on part du principe que les points vont toujours deux par deux
	return l2	

#on ne garde que les barres suffisamment longues pour représenter une barre de mesure ou de note
def garde_longues_barres_liste_de_liste(liste,taille_bv):
	l = []
	for elt in liste:
		l.append(garde_longues_barres(elt,[],taille_bv))
	return l

#une droite, une liste
def groupe_deux_points(l,l2):
	if (len(l) >= 2):
		l2.append([l[0],l[1]])
		groupe_deux_points(l[2:],l2)
	return l2
	
def groupe_deux_points_liste_de_liste(liste):
	l = []
	for elt in liste:
		l.append(groupe_deux_points(elt,[]))
	return l

#on supprime les barres qui sont trop proches (en tenant compte de leur ordonnée...
def supprime_barres_trop_proches(liste):
	l = []
	g = sorted(liste,key=lambda colonnes: colonnes[0])
	if len(g) >= 2:
		for i in range(len(g)-1):
			if (g[i][2] <= g[i+1][2]+entre_bv) and (g[i][2] >= g[i+1][2]-entre_bv):
				continue
			else:
				l.append(g[i])
		if (g[len(g)-2][2] > g[len(g)-1][2]+entre_bv) or (g[len(g)-2][2] < liste[len(g)-1][2]-entre_bv):
			l.append(g[len(g)-1])
	return l

#tracé des droites verticales
def trace_verticales(tupl):
	x = (tupl[2],tupl[2])
	y = (tupl[0],tupl[1])
	plt.plot(x,y,color = 'red')

def trace_verticales_liste(liste):
	for elt in liste:
		trace_verticales(elt)



#on retire les portées de l'image
def enleve_portees(img,soluce):
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if round(y*soluce[0]+soluce[1]) == round(x):
				img[x][y] = 1
				img[x-1][y] = 1
				img[x+1][y] = 1 #peu précis mais imprécision des droites détectées oblige
				
	return img

def enleve_portees_liste(img,liste):
	img1 = np.zeros(img.shape,np.int)
	img2 = np.zeros(img.shape,np.int)
	for elt in liste:
		img1 = enleve_portees(img,elt)
		img2 = union(img2,img1)
	return img2

#Dessine un structurant adapté pour l'ouverture en tout ou rien
def cree_structurant(rayon):
	h = int(3 + 4*rayon)
	l = int(7 + 2*rayon)
	a = np.zeros((h,l),int)
	#2 : indifférent, blanc ou noir
	for x in range(h):
		for y in range(l):
			a[x][y] = 2
	
	#1: points blancs
	"""for y in range(1+rayon):
		a[0][y] = 1	
		a[a.shape[0]-1][a.shape[1]-1-y] = 1"""
	"""for x in range(rayon):
		a[x][0] = 1	
		a[a.shape[0]-1-x][a.shape[1]-1] = 1"""
	for x in range(h):
		a[x][0] = 1
		a[x][l-1] = 1
	
	#0: points noirs
	for x in range(1+rayon,a.shape[0]-(1+rayon)):
		for y in range(a.shape[1]):
			a[x][a.shape[1]/2] = 0
	for x in range(1+rayon,a.shape[0]/2+1):
		for y in range(a.shape[1]/2-(x-(1+rayon)),1+a.shape[1]/2+(x-(1+rayon))):
			a[x][y] = 0
			a[a.shape[0]-x-1][y] = 0
	return a	

#détermine s'il y a une note à proximité de /collées à la barre verticale
def existe_note(img,ecart,i,j,seuil,coul):
	somme = 0
	rep = False
	ecart = int(round(ecart))
	for x in range(i-ecart,i):
		for y in range(j-ecart,j):
			if x < img.shape[0] and y < img.shape[1]:
				if img[x][y] == 0:
					somme = 1 + somme
				#plt.plot([j-ecart,j,j,j-ecart,j-ecart],[i-ecart-1,i-ecart-1,i-1,i-1,i-ecart-1])
	#si on remplit plus de 20% du carré "en bas"
	if somme*100 >= seuil*ecart*ecart:
		c1 = plt2.Circle(((2*j-ecart)/2,i),3*e/2,color=coul)
		plt2.gcf().gca().add_artist(c1)
		rep = True
	else:
		somme = 0
		for x in range(i-ecart,i):
			for y in range(j,j+ecart):
				if x < img.shape[0] and y < img.shape[1]:
					if img[x][y] == 0:
						somme = 1 + somme
					#plt.plot([j,j+ecart,j+ecart,j,j],[i-ecart+1,i-ecart+1,i+1,i+1,i-ecart+1])
		#si on remplit plus de 20% du carré "en haut"
		if somme*100 >= seuil*ecart*ecart:
			c1 = plt2.Circle(((2*j+ecart)/2,i),3*e/2,color=coul)
			plt2.gcf().gca().add_artist(c1)
			rep = True
	return rep

#pour chaque barre verticale identifiée, on regarde si c'est une note de musique
def existe_noire_img(img,liste,ecart):
	for elt in liste:
		elt.append(existe_note(img,ecart,elt[1],elt[2],pc_note,'g'))
		elt.append(existe_note(img,ecart,elt[0],elt[2],pc_note,'g'))
	return liste

#identifie une éventuelle croche en haut d'une barre verticale
def existe_croche_haut(img,ecart,i,j):
	somme = 0
	rep = 0
	ecart = int(round(ecart))
	e2 = int(round(ecart/2))
	for x in range(i,i+e2):
		for y in range(j-e2,j):
			if x < img.shape[0] and y < img.shape[1]:
				if img[x][y] == 0:
					somme = 1 + somme
	if somme*100 >= pc_cro*e2*e2:
		p = plt2.Rectangle((j,i),e2,e2,color='b')
		plt2.gcf().gca().add_artist(p)
		rep = 1
	else:
		somme = 0
		for x in range(i,i+e2):
			for y in range(j,j+e2):
				if x < img.shape[0] and y < img.shape[1]:
					if img[x][y] == 0:
						somme = 1 + somme
		if somme*100 >= pc_cro*e2*e2:
			p = plt2.Rectangle((j,i),e2,e2,color='b')
			plt2.gcf().gca().add_artist(p)
			rep = 1
	return rep

#identifie une éventuelle croche en bas d'une barre verticale
def existe_croche_bas(img,ecart,i,j):
	somme = 0
	rep = 0
	ecart = int(round(ecart))
	e2 = int(round(ecart/2))
	for x in range(i-e2,i):
		for y in range(j-e2,j):
			if x < img.shape[0] and y < img.shape[1]:
				if img[x][y] == 0:
					somme = 1 + somme
	if somme*100 >= pc_cro*e2*e2:
		p = plt2.Rectangle((j-e2,i-e2),e2,e2,color='b')
		plt2.gcf().gca().add_artist(p)
		rep = 1
	else:
		somme = 0
		for x in range(i-e2,i):
			for y in range(j,j+e2):
				if x < img.shape[0] and y < img.shape[1]:
					if img[x][y] == 0:
						somme = 1 + somme
		if somme*100 >= pc_cro*e2*e2:
			p = plt2.Rectangle((j-e2,i-e2),e2,e2,color='b')
			plt2.gcf().gca().add_artist(p)
			rep = 1
	return rep

#jusqu'à nbr_croches croches
def existe_autre_croche(img,liste,ecart):
	ecart = int(round(ecart))
	for elt in liste:
		if len(elt) > 5:
			if elt[5] != 0:
				for i in range(1,nbr_croches):
					elt[5] = (existe_croche_haut(img,ecart,elt[0]+i*ecart,elt[2]) or existe_croche_bas(img,ecart,elt[1]-i*ecart,elt[2])) + elt[5]
	return liste

#détermine suivant les résultats de l'existence de notes, l'existence de croches, de blanches ou de barres de mesure
def existe_croche_blanche_mesure(img,img2,liste,ecart):
	for elt in liste:
		#Si on a une noire en haut ou (exclusif) en bas
		if (not(elt[3]) and elt[4]) or (elt[3] and not(elt[4])):
			if elt[3]:
				elt.append(existe_croche_haut(img,ecart,elt[0],elt[2]))
			else:
				elt.append(existe_croche_bas(img,ecart,elt[1],elt[2]))
			#on regarde s'il y a d'autres croches
			liste = existe_autre_croche(img,liste,ecart)
			elt.extend([False,False])
			
		#s'il n'y a pas de noire
		elif (not(elt[3]) and not(elt[4])):
			#on met le nombre de croches à zéro
			elt.append(0)
			elt.append(existe_note(img2,ecart,elt[1],elt[2],pc_blan,'magenta'))
			elt.append(existe_note(img2,ecart,elt[0],elt[2],pc_blan,'magenta'))
			
			#c'est une barre de mesure (ni noire, ni blanche)
			if (not(elt[6]) and not(elt[7])):
				#elt.extend('m')
				x = [elt[2],elt[2]]
				y = [elt[0],elt[1]]
				plt.plot(x,y,'b')
	return liste
	
#----------------------------------------------------------
# Programme

#on passe en niveau de gris (marche seulement avec des .jpg)
img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)

#événement magique qui garde les noires et les croches
#cimg = cv2.medianBlur(img1,5)
#on passe à un seul channel
cimg = mh.colors.rgb2grey(cimg)

#on supprime les composantes connexes de trop petite taille
img12 = binary(img1)
img1 = areaclose(img12,2000)

#img22 = soustraction_img(img1,img2)
plt.imshow(img1)
plt.gray()
plt.show()

#on supprime tout ce qui n'est pas barre horizontal (à quelques degrés près)
a = 7
img2 = close(img1,seline(a,90))
img3 = close(img1,seline(a,89))
img4 = close(img1,seline(a,88))
img5 = close(img1,seline(a,87))
img6 = close(img1,seline(a,91))
img7 = close(img1,seline(a,92))
img8 = close(img1,seline(a,93))
img9 = close(img1,seline(a,94))

img = union(img2,img3,img4,img5,img6,img7,img8,img9)
plt.imshow(img)
plt.show()

#Vachement long si on trace les points rouges
l1 = trouve_noir_matrice(img)
l2 = les_milieux_liste_de_liste(l1)
l4 = groupe_cinq_points_liste_de_liste(l2)
l4b = ajoute_abscisse(l4)
l4t = liste_sans_liste_vide(l4b)
l5 = split_listes(l4t)

#On a les coordonnées des droites (5) de chaque portée (n)
lprem = premiere_coordonnee_liste_de_liste(l5)
lprem = separe_les_portees(lprem,[])
lsec = deuxieme_coordonnee_liste_de_liste(l5)
lsec = separe_les_portees(lsec,[])
lter = troisieme_coordonnee_liste_de_liste(l5)
lter = separe_les_portees(lter,[])
lqua = quatrieme_coordonnee_liste_de_liste(l5)
lqua = separe_les_portees(lqua,[])
lcin = cinquieme_coordonnee_liste_de_liste(l5)
lcin = separe_les_portees(lcin,[])


#on applique la méthode des moindres carrées pour trouver la droite de la portée

#On calcule les coefficients (A, B, C, D, E, F)
abcdef_prem = calcul_abcdef_plusieurs_listes(lprem)
abcdef_sec = calcul_abcdef_plusieurs_listes(lsec)
abcdef_ter = calcul_abcdef_plusieurs_listes(lter)
abcdef_qua = calcul_abcdef_plusieurs_listes(lqua)
abcdef_cin = calcul_abcdef_plusieurs_listes(lcin)

#On calcule les solutions (b,c) : (pente,ordonnée à l'origine)
solprem = solution_liste(abcdef_prem)
solsec = solution_liste(abcdef_sec)
solter = solution_liste(abcdef_ter)
solqua = solution_liste(abcdef_qua)
solcin = solution_liste(abcdef_cin)

#liste des listes de couples solutions
tab = [solprem,solsec,solter,solqua,solcin]

#A partir des équations, on trace les droites de chaque portée
tracer_droite_liste(solprem,img1)
tracer_droite_liste(solsec,img1)
tracer_droite_liste(solter,img1)
tracer_droite_liste(solqua,img1)
tracer_droite_liste(solcin,img1)

#On affiche l'image et les droites
plt.imshow(img1)
plt.gray()
plt.show()

moy = moyenne_pentes(tab)
e0 = ecart_moyen(tab)

#PERTINENT ?
#changement de repère
img2 = changement_repere(img1,moy)

#nouveau tracé des droites
tab2 = changement_de_repere_tableau(img1,tab,moy)
tracer_droite_hori_liste(tab2,img2)

plt.imshow(img2)
plt.show()

#détection des barres verticales
#on ne garde que les barres > 3*écart entre les lignes de portée
img3 = close(img2,seline(3*e0))
plt.imshow(img3)
plt.show()

v1 = trouve_barres_verticales(img3)
#v2 = garde_longues_barres_liste_de_liste(v1,3*e0)
v3 = groupe_deux_points_liste_de_liste(v1)
v4 = ajoute_abscisse(v3)
v4b = liste_sans_liste_vide(v4)
v5 = split_listes(v4b)
v6 = supprime_barres_trop_proches(v5)

trace_verticales_liste(v6)
tracer_droite_hori_liste(tab2,img2)
plt.imshow(img2)
plt.show()
	
#détection des notes

#On retire les portées /!\ ici on le fait après le changement de repère (bonne idée ?)
img11 = enleve_portees_liste(img1,solprem)
img12 = enleve_portees_liste(img11,solsec)
img13 = enleve_portees_liste(img12,solter)
img14 = enleve_portees_liste(img13,solqua)
img15 = enleve_portees_liste(img14,solcin)

plt.imshow(img15)
plt.show()

#recolle les morceaux de notes
img3 = open(img15,seline(3.5,0))
plt.imshow(img3)
plt.show()

#partition5 : close/sedisk(6) /seline(20)
#partition2 : close/sedisk(1) /seline(20)
#partition8 : close/sedisk(1) /seline(20)

#image ne contenant que (ou presque) les notes
img51 = close(img3,sedisk(1))
#on enleve les barres horizontales possiblement restantes
b = 20
img52 = close(img51,seline(b,90))
img53 = close(img51,seline(b,89))
img54 = close(img51,seline(b,88))
img55 = close(img51,seline(b,91))
img56 = close(img51,seline(b,92))
img57 = union(img52,img53,img54,img55,img56)

img5 = soustraction_img(img51,img57)
plt.imshow(img5)
plt.show()

#forme des listes [ordonnée1,ordonnée2,abscisse // noire en bas ?, noire en haut ? // nbr de croches // blanche en bas ?, blanche en haut ? // 'm']
trace_verticales_liste(v6)
v7 = existe_noire_img(img5,v6,e0)
#img3 sert à détecter les croches, img2 sert à détecter les blanches
v8 = existe_croche_blanche_mesure(img3,img2,v7,e0)

tracer_droite_liste(solprem,img1)
tracer_droite_liste(solsec,img1)
tracer_droite_liste(solter,img1)
tracer_droite_liste(solqua,img1)
tracer_droite_liste(solcin,img1)

plt.imshow(img3)
plt.show()
