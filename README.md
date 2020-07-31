# exploradome_tangram
Tangram form detection from live video stream

## Approach

TBD  
Keywords : Hu Moments, distances

## Testing

### Unit Tests

We test each function individually

### Integration Tests

#### Static tests
We compare an image with our dataset. We need several images by class.

#### Video tests
We compare a video with our dataset. We need different sequences with a clear solution


## TODOS

### Récupération des caractéristiques des images
- [x] fonction pour récupérer les Hu moments
- [x] récupération des Hu moments dataset tangrams
- [x] processing des images pour récupérer leur contour
- [x] récupérer le nombre de sommets dans une image
- [x] utiliser les sommets pour déterminer si on peut calculer les probabilités - non significatif
- [ ] utiliser l'aire globale du tangram pour améliorer les prédictions - Bastien 
- [x] écriture des fonctions de calcul de distances
- [x] convertir distances en probabilités - Gauthier
- [ ] calcul du périmètre pour améliorer les prédictions - Nicolas
- [ ] identifier le plateau de jeu - Nicolas
- [ ] recherche de fonctions pour le traitement des mains - Bastien
- [ ] récupération des formes géométriques - Gautier

### Prédictions
- [x] affichage des prédictions en temps réel

### Tests
- [ ] tests unitaires - Renata
- [x] stratégie de test
- [x] optimisation du code
- [ ] découpage des videos - Nohossat
- [ ] tests d'intégration - Renata & Nohossat
- [ ] récupération des métriques - Renata & Nohossat

### Version control
- [ ] ecriture du readme - TBD
- [x] ré-organisation des fichiers dans le GitHub - Nohossat

### Dataset avec les tangrams

[Dataset](https://drive.google.com/drive/folders/1pmuPaserBOOIrdrdmM8uy592v4ylJlHx?usp=sharing)
