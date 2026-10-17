import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane

// =======================
// CONFIG
// =======================
def boundaryCSV = new File(buildFilePath(
        PROJECT_BASE_DIR, 'output', 'glomeruli_boundaries.csv'
))
def centroidCSV = new File(buildFilePath(
        PROJECT_BASE_DIR, 'output', 'lumen_centroids.csv'
))
def metricsCSV = new File(buildFilePath(
        PROJECT_BASE_DIR, 'output', 'artery_metrics.csv'
))
def raysCSV = new File(buildFilePath(
        PROJECT_BASE_DIR, 'output', 'ray_thickness_segments.csv'
))

def plane = ImagePlane.getDefaultPlane()

setBatchProjectAndImage(getProject(), getCurrentImageData())

// =======================
// DEBUG LOGS (ALWAYS FIRST)
// =======================
println "=== SCRIPT STARTED ==="
println "PROJECT_BASE_DIR = ${PROJECT_BASE_DIR}"
println "Boundary CSV = ${boundaryCSV.getAbsolutePath()}"
println "Centroid CSV = ${centroidCSV.getAbsolutePath()}"
println "Metrics CSV = ${metricsCSV.getAbsolutePath()}"
println "Rays CSV = ${raysCSV.getAbsolutePath()}"
println "Boundary CSV exists = ${boundaryCSV.exists()}"
println "Centroid CSV exists = ${centroidCSV.exists()}"
println "Metrics CSV exists = ${metricsCSV.exists()}"
println "Rays CSV exists = ${raysCSV.exists()}"

if (!boundaryCSV.exists() || !centroidCSV.exists()) {
    println "ERROR: Required CSV files not found in project/output/"
    return
}

// =======================
// LOAD CENTROID INFO
// =======================
def lumenInfo = [:]   // key = annotation_id

centroidCSV.readLines().eachWithIndex { line, i ->
    if (i == 0) return

    def p = line.split(',')
    lumenInfo[p[0]] = [
        cx: p[2].toDouble(),
        cy: p[3].toDouble(),
        category: p[1]
    ]
}

println "Loaded ${lumenInfo.size()} centroids"

// =======================
// LOAD METRICS INFO (OPTIONAL)
// =======================
def metricsInfo = [:]  // key = annotation_id
if (metricsCSV.exists()) {
    metricsCSV.readLines().eachWithIndex { line, i ->
        if (i == 0) return
        def p = line.split(',')
        if (p.size() < 14) return
        metricsInfo[p[0]] = [
            homogeneity: p[1],
            shape_similarity: parseDoubleSafe(p[2]),
            concentricity_index: parseDoubleSafe(p[3]),
            lumen_count: parseIntSafe(p[4]),
            multi_lumen_complexity: parseDoubleSafe(p[5]),
            wall_thickness_mean: parseDoubleSafe(p[6]),
            wall_thickness_cv: parseDoubleSafe(p[7]),
            narrowing_index: parseDoubleSafe(p[8]),
            flag_level: p[10],
            risk_score: parseIntSafe(p[11]),
            aif_percent: parseDoubleSafe(p[12]),
            severity_grade: parseIntSafe(p[13]),
        ]
    }
}
println "Loaded ${metricsInfo.size()} artery metrics rows"

// =======================
// LOAD RAY SEGMENTS (OPTIONAL)
// =======================
int plottedRays = 0
if (raysCSV.exists()) {
    raysCSV.readLines().eachWithIndex { line, i ->
        if (i == 0) return
        def p = line.split(',')
        if (p.size() < 8) return
        double x1 = parseDoubleSafe(p[2])
        double y1 = parseDoubleSafe(p[3])
        double x2 = parseDoubleSafe(p[4])
        double y2 = parseDoubleSafe(p[5])
        double tNorm = parseDoubleSafe(p[7])

        def roi = ROIs.createLineROI(x1, y1, x2, y2, plane)
        def obj = PathObjects.createAnnotationObject(roi)
        def c = colorForThicknessNorm(tNorm)
        obj.setColor(c[0] as int, c[1] as int, c[2] as int)
        obj.setName(String.format("Ray tNorm=%.2f", tNorm))
        addObject(obj)
        plottedRays++
    }
}
println "Plotted ${plottedRays} wall-thickness rays"

// =======================
// LOAD & PLOT BOUNDARIES
// =======================
def currentId = null
def currentCategory = null
def xs = []
def ys = []
int plotted = 0

boundaryCSV.readLines().eachWithIndex { line, i ->
    if (i == 0) return

    def p = line.split(',')
    def annotationId = p[0]
    def category = p[1]
    double x = p[3].toDouble()
    double y = p[4].toDouble()

    if (currentId != null && annotationId != currentId) {
        plotPolygon(currentId, currentCategory, xs, ys, plane, lumenInfo, metricsInfo)
        plotted++
        xs.clear()
        ys.clear()
    }

    currentId = annotationId
    currentCategory = category
    xs << x
    ys << y
}

// Plot last polygon
if (xs.size() > 2) {
    plotPolygon(currentId, currentCategory, xs, ys, plane, lumenInfo, metricsInfo)
    plotted++
}

println "SUCCESS: Plotted ${plotted} lumen polygons"

// =======================
// FUNCTION
// =======================
def plotPolygon(String annotationId, String category,
                List xs, List ys,
                ImagePlane plane, Map lumenInfo, Map metricsInfo) {

    def roi = ROIs.createPolygonROI(xs as double[], ys as double[], plane)
    def obj = PathObjects.createAnnotationObject(roi)
    def metric = metricsInfo[annotationId]

    // Prioritize review level colors if available; fallback to homogeneity palette.
    if (metric != null && metric.flag_level == 'urgent_review') {
        obj.setColor(255, 0, 0)   // RED
    } else if (metric != null && metric.flag_level == 'review_recommended') {
        obj.setColor(255, 165, 0) // ORANGE
    } else if (category == 'homogeneous' || category == 'normal') {
        obj.setColor(0, 0, 255)   // BLUE
    } else if (category == 'heterogeneous' || category == 'abnormal') {
        obj.setColor(255, 80, 0)  // RED-ORANGE
    } else {
        obj.setColor(255, 255, 0) // YELLOW (uncertain/no_lumen/other)
    }

    if (metric != null) {
        def ci = metric.concentricity_index
        def ml = metric.lumen_count
        def wt = metric.wall_thickness_cv
        def ni = metric.narrowing_index
        def sev = metric.severity_grade
        def risk = metric.risk_score
        obj.setName(String.format("A%s H=%s CI=%.2f ML=%d WTcv=%.2f NI=%.2f G%d R%d",
                annotationId, category, ci, ml, wt, ni, sev, risk))
    } else {
        obj.setName("A${annotationId} H=${category}")
    }

    addObject(obj)

    // PLOT CENTROID
    if (lumenInfo.containsKey(annotationId)) {
        def info = lumenInfo[annotationId]

        def cRoi = ROIs.createPointsROI(
            [info.cx] as double[],
            [info.cy] as double[],
            plane
        )

        def cObj = PathObjects.createAnnotationObject(cRoi)
        cObj.setColor(0, 255, 0)  // GREEN centroid
        cObj.setName("Lumen-Centroid-${annotationId}")
        addObject(cObj)
    }
}

def parseDoubleSafe(String s) {
    try {
        return s?.trim() ? s.toDouble() : 0.0
    } catch (Exception ignored) {
        return 0.0
    }
}

def parseIntSafe(String s) {
    try {
        return s?.trim() ? s.toInteger() : 0
    } catch (Exception ignored) {
        return 0
    }
}

def colorForThicknessNorm(double tNorm) {
    if (tNorm >= 1.30) return [255, 0, 0]      // thick/high variation
    if (tNorm >= 1.10) return [255, 165, 0]    // mildly elevated
    if (tNorm <= 0.75) return [0, 0, 255]      // thinner
    return [0, 200, 0]                         // baseline
}

def entry = getProjectEntry()
def imageData = getCurrentImageData()
if (entry != null && imageData != null) {
    entry.saveImageData(imageData)
    println "PROJECT IMAGE DATA SAVED"
} else {
    println "WARNING: Could not save project image data (missing project entry or image data)"
}