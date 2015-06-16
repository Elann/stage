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
import sys
import warnings

#----------------------------------------------------------
# Importation de l'image

img0 = cv2.imread('images/partition2.jpg',0)

# si problème avec la fonction qui grise :  as_grey=True (ne garantit pas des entiers)

#empêche les warning (apparus par magie) quand on ferme la fenêtre de pyplot
warnings.simplefilter("ignore")

#----------------------------------------------------------
# Variables globales empiriques

#longueur max d'un intervalle de noir représentant une portée
seuil_portee = 2
#sert au seuillage
seuil_noir = 100 
#marge autorisée pour les distances entre les lignes de la portée
delta_portee = 2
#marge autorisée entre les ordonnées des points d'une même ligne de portée
delta_dist = 5
#Nombre de pixels minimum qu'il doit y avoir entre deux barres verticales
entre_bv = 3
#pourcentage de remplissage du carré pour qu'on ait une note
#pc_note = e0
#pourcentage de remplissage du carré pour qu'on ait une croche
#pc_cro = 45
#pourcentage de remplissage du carré pour qu'on ait une blanche
#pc_blan = e0
#Nombre maximal de croches sur une même note
nbr_croches = 2


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

def imgbooltoint(img):
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if img[x][y]:
				img[x][y] = 1
			else:
				img[x][y] = 0
	return img
				
#Teste si 5 ordonnées peuvent former une portée
def test_cinq(a,b,c,d,e):
	if (b-a) < delta_portee+(c-b) and (c-b) < delta_portee+(d-c) and (d-c) < delta_portee+(e-d) and (e-d) < delta_portee+(b-a):
		rep = 1
	else:
		rep = 0
	return rep

#Groupe les résultats de la projection en portées
def groupe_portee(l,l2):
	if len(l) != 0:
		try:
			len(l)%5 != 0
			if test_cinq(l[0][1],l[1][1],l[2][1],l[3][1],l[4][1]) == 1:
				l2.append(l[:5])
				groupe_portee(l[5:],l2)
			else:
				groupe_portee(l[1:],l2)
		except:
			print "pas de portées correctes détectées"
			sys.exit(1)
	return l2

def projection_lignes(img):
	a = np.zeros(img.shape[0],np.int)
	l = []
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] == 0:
				a[i] = a[i] + 1
	#on ajoute les ordonnées
	for i in range(len(a)):
		l.append((a[i],i))
	return l

def maxi_locaux(a,img):
	l = []
	for i in range(2,len(a)-2):
		if a[i][0] > a[i-2][0] and a[i][0] > a[i-1][0] and a[i][0] >= a[i+1][0] and a[i][0] > a[i+2][0] and (3*a[i][0] > 2*img.shape[1]):
			l.append(a[i])
	#mod = len(l)%5
	#if mod != 0:
	#	for i in range(mod):
	#		l.remove(min(l))
	return l
	
def garde_ordonnees(liste):
	l2 = []
	for j in range(5):
		l = []
		for i in range(len(liste)):
			l.append(liste[i][j][1])
		l2.append(l)
	return l2

def ecart_moyen(liste):
	somme = 0
	compt = 0
	for i in range(len(liste)-1):
		for j in range(len(liste[i])):
			somme = abs(liste[i][j]-liste[i+1][j]) + somme
			compt = 1 + compt
	return somme/compt	

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

img1 = cv2.adaptiveThreshold(img0,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)

"""
#on passe en niveau de gris (marche seulement avec des .jpg)
img0 = cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)

#on passe à un seul channel
img1 = mh.colors.rgb2grey(img0,np.int)

h = histogram(img1)
g = cv2.EM.predict(h)
print g
plt.imshow(g)
plt.show()
print h

h = hmin(img1)
g = hmax(img1)
img12 = threshad(img1,h,g)

img12 = binary(img1,200)
"""

plt.imshow(img1)
plt.gray()
plt.show()

#on supprime les composantes connexes de trop petite taille
img1 = areaclose(img1,2000)

plt.imshow(img1)
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

#On projette sur l'axe des ordonnées pour trouver les portées
l = projection_lignes(img)
l1 = maxi_locaux(l,img)
l2 = groupe_portee(l1,[])
#Ordonnées à l'origine de toutes les portées [[ligne1],[ligne2],...,[ligne5]]
l3 = garde_ordonnees(l2)
print l3

#on affiche l'image et les droites
tracer_droite_hori_liste(l3,img)
plt.imshow(img)
plt.show()

try:
	e0 = ecart_moyen(l3)
except IndexError:
	print "erreur lors de la détection des portées"
	sys.exit(1)

#détection des barres verticales
#on ne garde que les barres > 3*écart entre les lignes de portée
img3 = close(img1,seline(3*e0))
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
tracer_droite_hori_liste(l3,img1)
plt.imshow(img1)
plt.show()

#événement magique qui garde les noires et les croches
cimg = cv2.medianBlur(img0,5)

#on passe en binaire (fait n'importe quoi avec le threshold Gaussien)
cimg = binary(cimg,100)
cimg = imgbooltoint(cimg)

trace_verticales_liste(v6)

plt.imshow(cimg)
plt.show()
print cimg
#détection des notes

pc_note = e0 #5*(e0-1)/4
pc_blan = 50
pc_cro = e0

#on enleve les barres ~horizontales possiblement restantes
b = 2*e0 #taille suivant les images !
img52 = close(cimg,seline(b,90))
img53 = close(cimg,seline(b,89))
img54 = close(cimg,seline(b,88))
img55 = close(cimg,seline(b,91))
img56 = close(cimg,seline(b,92))
img57 = close(cimg,seline(b,93))
img58 = close(cimg,seline(b,87))
img59 = close(cimg,seline(b,86))
img60 = close(cimg,seline(b,94))
img61 = close(cimg,seline(b,0))
img62 = union(img52,img53,img54,img55,img56,img57,img58,img59,img60)

img5 = soustraction_img(cimg,img62)
img5 = soustraction_img(img5,img61)
plt.imshow(img5)
plt.show()

#forme des listes [ordonnée1,ordonnée2,abscisse // noire en bas ?, noire en haut ? // nbr de croches // blanche en bas ?, blanche en haut ? // 'm']
trace_verticales_liste(v6)
v7 = existe_noire_img(img5,v6,e0)

#cimg sert à détecter les croches, img1 sert à détecter les blanches
v8 = existe_croche_blanche_mesure(cimg,img1,v7,e0)

tracer_droite_hori_liste(l3,img3)
plt.imshow(img3)
plt.show()
