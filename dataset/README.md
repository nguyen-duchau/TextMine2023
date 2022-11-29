# Challenge_TextMine_2023 :  "Reconnaissance d’entités d’intérêts dans les signatures d’e-mails"

Le 21 octobre 2022, l'association Extraction et Gestion des Connaissances (EGC) a lancé le groupe de travail TextMine. Dans le cadre de ce groupe de travail, un objectif est de confronter l'état de l'art scientifique aux problèmes de text mining rencontrés par des industriels. Sous la forme de défis, le groupe de travail propose des jeux de données inédits et les partage avec la communauté scientifique. 
Le premier défi du groupe de travail TextMine a été lancé le 21 octobre en étroite collaboration avec la société Emvista, éditrice de logiciels fondés sur des technologies du Traitement Automatique du Langage Naturel, qui a fourni une partie des données. En particulier, la société s'intéresse à la structuration des informations véhiculées dans les e-mails.

Le défi proposé se focalise sur la reconnaissance d'entités d'intérêts dans les signatures d'e-mails dans le but de structurer l'information et de la stocker en base de données (par exemple un système de gestion de la relation client).


# Participer
1/ Se signaler auprès de textmine@emvista.com afin d'être notifié des éventuelles mises à jour et autres informations  
2/ Accès au github pour prendre connaissance du défi et acquérir les jeux de données  
3/ Envoyer le fichier des résultats par e-mail à textmine@emvista.com (3 envois possibles au maximum) avant le 10 janvier 2023  
4/ Notification du résultat par les organisateurs après chaque envoi  
5/ Publication du résultat à l’atelier TextMine (17 janvier 2023) et attribution du prix  

# Data

Description des données sur le PDF "Défi TextMine’23 - Reconnaissance d’entités d’intérêts dans les signatures d’e-mails"

Face à l'absence de données annotées, le groupe de travail a adopté trois stratégies de génération de jeu de données qui ont conduit à la création de trois jeux de données :
- Jeu de données authentique (JDA dans la suite) : un jeu de données composé de signatures authentiques pseudonymisées ; non disponible pour l'instant (jeu de test sur lequel seront évalués les participants au défi)
- Jeu de données réaliste (JDR dans la suite) : un jeu de données composé de signatures construites manuellement par la société Isahit, plateforme de labellisation éthique des données pour l'IA ; ce jeu de données contient des signatures réalistes, c'est-à-dire proches des signatures authentiques observées, mais non authentiques (elles n'ont jamais été utilisées dans des échanges d'e-mails) ; disponibles ici : https://github.com/Emvista/Challenge_TextMine_2023/blob/main/JDR.json
- Jeu de données factice (JDF dans la suite) : un jeu de donnés composé de signatures créées automatiquement à partir d'une API de génération de fausses identités ; disponible ici : https://github.com/Emvista/Challenge_TextMine_2023/blob/main/JDF.json

# Citer

K. Cousot, C. Lopez, P. Cuxac, V. Lemaire (2023) Défi TextMine’23 - Reconnaissance d’entités d’intérêts dans les signatures d’e-mails. _Actes de l'atelier TextMine'23_, p. à paraître, Conférence Extraction et Gestion des Connaissances 2023 (EGC'23), Lyon. https://github.com/Emvista/Challenge_TextMine_2023/blob/main/Challenge_TextMine_2023.pdf
