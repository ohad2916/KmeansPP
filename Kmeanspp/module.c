#define PY_SSIZE_T_CLEAN
#include <Python.h>


double euc_d(double* p, double* q, size_t dim) {
    double d_sqrd_sum = 0;
    size_t i = 0;
    while (i < dim) {
        d_sqrd_sum += pow(p[i] - q[i], 2);
        i++;
    }
    return sqrt(d_sqrd_sum);
}
int free_memory(double* a, double* b, double* c, double** data, double** mean, double** curr) {
    free(a);
    free(b);
    free(c);
    free(data);
    free(mean);
    free(curr);
    return 0;
}
static PyObject* fit(PyObject* self, PyObject* args) {
    PyObject* data_lst;
    PyObject* centroid_lst;
    PyObject* point;
    size_t iter;
    double epsilon;
    if (!PyArg_ParseTuple(args, "OOnd", &data_lst,&centroid_lst,&iter,&epsilon)) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        return NULL;
    }
    /*data_lst = dataframe */
    size_t no_points = PyObject_Length(data_lst);
    point = PyList_GetItem(data_lst, 0);
    size_t dimension = PyObject_Length(point); 
    size_t centroid_size = PyObject_Length(centroid_lst);

    //debuging here
    /*for (size_t i = 0; i < no_points; i++){
        point = PyList_GetItem(data_lst, i);
        for (size_t j = 0; j < dimension; j++) {
            printf("%f,", PyFloat_AsDouble(PyList_GetItem(point, j)));
        }
        printf("\n");
    }*/

    /*convert python data-list to a c double array*/
    double* p = NULL;
    double **data = NULL;
    double* b = NULL;
    double** cluster_mean = NULL;
    double* c = NULL;
    double** new_cluster = NULL;
    data = calloc(no_points, sizeof(double*));
    if (!data) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        free_memory(b, c, p, new_cluster, cluster_mean, data);
        return NULL;
    }
    p = NULL;
    p = (double*)calloc(no_points * dimension, sizeof(double));
    if (!p) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        free_memory(b, c, p, new_cluster, cluster_mean, data);
        return NULL;
    }
    for (size_t i = 0; i < no_points; i++) {
        data[i] = p + i * dimension;
    }

    for (size_t i = 0; i < no_points; i++) {
        point = PyList_GetItem(data_lst, i);
        for (size_t j = 0; j < dimension; j++) {
            data[i][j] = PyFloat_AsDouble(PyList_GetItem(point, j));
            //printf("%f,", data[i][j]);
        }
    }
    /*convert current cluster list to a c array*/
    b = calloc(centroid_size * dimension, sizeof(double));
    cluster_mean = calloc(centroid_size, sizeof(double*));
    if (!b || !cluster_mean) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        free_memory(b, c, p, new_cluster, cluster_mean, data);
        return NULL;
    }
    for (size_t i = 0; i < centroid_size; i++) {
        cluster_mean[i] = b + i * (dimension);
    }

    for (size_t i = 0; i < centroid_size; i++)
    {
        point = PyList_GetItem(centroid_lst, i);
        for (size_t j = 0; j < dimension; j++){
            cluster_mean[i][j] = PyFloat_AsDouble(PyList_GetItem(point, j));
        }
    }
    /*main algorithm*/
    double** curr_X = data;
    /*allocate temporary clusters to decide convergence*/
    c = calloc(centroid_size * (dimension + 1), sizeof(double));
    new_cluster = calloc(centroid_size, sizeof(double*));
    if (!c || !new_cluster) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        free_memory(b, c, p, new_cluster, cluster_mean, data);
        return NULL;
    }
    for (size_t i = 0; i < centroid_size; i++) {
        new_cluster[i] = c + i * (dimension + 1);
    }

    size_t i = 0;
    int converged = 0;
    while (i < iter && !converged)
    {
        /*zero out new cluster array*/
        memset(c, 0, centroid_size * (dimension + 1) * sizeof(double));
        /*decide closest cluster against the original and update it with new Xi*/
        for (size_t m = 0; m < no_points; m++) {
            size_t min_cluster_index = 0;
            double min_value = INT32_MAX;
            for (size_t j = 0; j < centroid_size; j++)
            {
                double curr_euc_d = euc_d(cluster_mean[j], *(curr_X + m), dimension);
                if (curr_euc_d < min_value)
                {
                    min_value = curr_euc_d;
                    min_cluster_index = j;
                }
            }
            /*updating new cluster, just adding for now. divide later.*/
            double* min_cluster = new_cluster[min_cluster_index];
            for (size_t j = 0; j < dimension; j++)
            {
                min_cluster[j] += curr_X[m][j];
            }
            min_cluster[dimension]++;
        }
        /*calculate the actual means*/
        for (size_t j = 0; j < centroid_size; j++) {
            for (size_t m = 0; m < dimension; m++) {
                new_cluster[j][m] /= new_cluster[j][dimension];
            }
        }

        /*decide convegerence*/
        double max_Duk = 0;
        double curr_Muk = 0;
        for (size_t j = 0; j < centroid_size; j++) {
            curr_Muk = euc_d(cluster_mean[j], new_cluster[j], dimension);
            if (curr_Muk > max_Duk)
                max_Duk = curr_Muk;
        }
        if (max_Duk <= epsilon) {
            /*print statement for debugging*/
            /*printf("Converged after: %d iterations\n", (int)i + 1);*/
            converged = 1;
        }
        i++;
        /*copy new cluster to old ones*/
        for (size_t j = 0; j < centroid_size; j++)
        {
            for (size_t m = 0; m < dimension; m++)
            {
                cluster_mean[j][m] = new_cluster[j][m];
            }
        }
    }
    PyObject* py_centroids = PyList_New(centroid_size);
    for (size_t m = 0; m < centroid_size; m++)
    {
        PyObject* m_cluster = PyList_New(dimension); 
        PyList_SetItem(py_centroids, m, m_cluster);        //raises exception
        for (size_t j = 0; j < dimension; j++)
        {
            PyList_SetItem(m_cluster, j,PyFloat_FromDouble(cluster_mean[m][j]));
        }
    }
    free_memory(b, c, p, new_cluster, cluster_mean, data);
	return py_centroids;
}

static PyMethodDef methods[] = {
    {"fit",                   /* the Python method name that will be used */
      (PyCFunction)fit, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* flags indicating parameters
accepted for this function */
      PyDoc_STR("takes 2 python lists, max iteration value, Convergence value")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};

static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject* m;
    m = PyModule_Create(&kmeansmodule);
    if (!m) {
        return NULL;
    }
    return m;
}