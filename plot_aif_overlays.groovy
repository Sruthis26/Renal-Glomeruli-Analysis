import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane

// =======================
// CONFIG
// =======================
def arteryCSV = new File(buildFilePath(
        PROJECT_BASE_DIR, 'output', 'aif_artery_boundaries.csv'
))
def lumenCSV = new File(buildFilePath(
        PROJECT_BASE_DIR, 'output', 'aif_lumen_boundaries.csv'
))
def centroidsCSV = new File(buildFilePath(
        PROJECT_BASE_DIR, 'output', 'aif_centroids.csv'
))
def circlesCSV = new File(buildFilePath(
        PROJECT_BASE_DIR, 'output', 'aif_circles.csv'
))

def plane = ImagePlane.getDefaultPlane()

setBatchProjectAndImage(getProject(), getCurrentImageData())

println "=== AIF SCRIPT STARTED ==="
println "PROJECT_BASE_DIR = ${PROJECT_BASE_DIR}"
println "Artery CSV = ${arteryCSV.getAbsolutePath()} exists=${arteryCSV.exists()}"
println "Lumen CSV = ${lumenCSV.getAbsolutePath()} exists=${lumenCSV.exists()}"
println "Centroids CSV = ${centroidsCSV.getAbsolutePath()} exists=${centroidsCSV.exists()}"
println "Circles CSV = ${circlesCSV.getAbsolutePath()} exists=${circlesCSV.exists()}"

if (!arteryCSV.exists() || !lumenCSV.exists() || !centroidsCSV.exists() || !circlesCSV.exists()) {
    println "ERROR: Required AIF CSV files not found in project/output/"
    return
}

// =======================
// LOAD CENTROIDS
// =======================
def centroids = []
centroidsCSV.readLines().eachWithIndex { line, i ->
    if (i == 0) return
    def p = line.split(',')
    centroids << [
        id: p[0],
        type: p[1],
        category: p[2],
        cx: p[3].toDouble(),
        cy: p[4].toDouble(),
        aif: p[5].toDouble()
    ]
}

// =======================
// LOAD CIRCLES
// =======================
def circles = []
circlesCSV.readLines().eachWithIndex { line, i ->
    if (i == 0) return
    def p = line.split(',')
    circles << [
        id: p[0],
        circleType: p[1],
        category: p[2],
        cx: p[3].toDouble(),
        cy: p[4].toDouble(),
        r: p[5].toDouble()
    ]
}

// =======================
// PLOT ARTERY POLYGONS (RED)
// =======================
def currentId = null
def currentCategory = null
def xs = []
def ys = []
int plottedArteries = 0

arteryCSV.readLines().eachWithIndex { line, i ->
    if (i == 0) return
    def p = line.split(',')
    def annotationId = p[0]
    def category = p[1]
    double x = p[3].toDouble()
    double y = p[4].toDouble()

    if (currentId != null && annotationId != currentId) {
        plotPolygon(currentId, currentCategory, xs, ys, plane, "AIF-Artery")
        plottedArteries++
        xs.clear()
        ys.clear()
    }

    currentId = annotationId
    currentCategory = category
    xs << x
    ys << y
}

if (xs.size() > 2) {
    plotPolygon(currentId, currentCategory, xs, ys, plane, "AIF-Artery")
    plottedArteries++
}

println "Plotted ${plottedArteries} artery polygons"

// =======================
// PLOT LUMEN POLYGONS (BLUE)
// =======================
currentId = null
currentCategory = null
xs = []
ys = []
int plottedLumens = 0

lumenCSV.readLines().eachWithIndex { line, i ->
    if (i == 0) return
    def p = line.split(',')
    def annotationId = p[0]
    def category = p[1]
    double x = p[3].toDouble()
    double y = p[4].toDouble()

    if (currentId != null && annotationId != currentId) {
        plotPolygon(currentId, currentCategory, xs, ys, plane, "AIF-Lumen")
        plottedLumens++
        xs.clear()
        ys.clear()
    }

    currentId = annotationId
    currentCategory = category
    xs << x
    ys << y
}

if (xs.size() > 2) {
    plotPolygon(currentId, currentCategory, xs, ys, plane, "AIF-Lumen")
    plottedLumens++
}

println "Plotted ${plottedLumens} lumen polygons"

// =======================
// PLOT CIRCLES (GREEN/YELLOW)
// =======================
circles.each { c ->
    double x = c.cx - c.r
    double y = c.cy - c.r
    double w = c.r * 2
    double h = c.r * 2
    def roi = ROIs.createEllipseROI(x, y, w, h, plane)
    def obj = PathObjects.createAnnotationObject(roi)
    if (c.circleType == 'outer') {
        obj.setColor(0, 255, 0) // green
    } else {
        obj.setColor(255, 255, 0) // yellow
    }
    obj.setName("AIF-${c.circleType}")
    addObject(obj)
}

println "Plotted ${circles.size()} circles"

// =======================
// PLOT CENTROIDS (RED/BLUE)
// =======================
centroids.each { c ->
    def roi = ROIs.createPointsROI([c.cx] as double[], [c.cy] as double[], plane)
    def obj = PathObjects.createAnnotationObject(roi)
    if (c.type == 'artery') {
        obj.setColor(255, 0, 0)
    } else {
        obj.setColor(0, 0, 255)
    }
    obj.setName(String.format("AIF %.2f%% - %s", Math.abs(c.aif), c.category))
    addObject(obj)
}

def entry = getProjectEntry()
def imageData = getCurrentImageData()
if (entry != null && imageData != null) {
    entry.saveImageData(imageData)
    println "PROJECT IMAGE DATA SAVED"
} else {
    println "WARNING: Could not save project image data (missing project entry or image data)"
}

def colorForCategory(String category) {
    if (category == "high_narrowing") return [255, 0, 0]
    if (category == "moderate_narrowing") return [255, 165, 0]
    if (category == "low_narrowing") return [0, 220, 0]
    return [0, 0, 255]
}

def plotPolygon(String annotationId, String category, List xs, List ys, ImagePlane plane, String namePrefix) {
    def roi = ROIs.createPolygonROI(xs as double[], ys as double[], plane)
    def obj = PathObjects.createAnnotationObject(roi)
    def c = colorForCategory(category)
    obj.setColor(c[0] as int, c[1] as int, c[2] as int)
    obj.setName("${namePrefix}-${annotationId}-${category}")
    addObject(obj)
}