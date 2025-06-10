#ifndef MATRIX_H_
#define MATRIX_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include "util.h"

typedef struct Matrix 
{
	float* data;
	size_t num_rows;
	size_t num_cols;
	size_t stride;
}Matrix;

#define M_ELEM_AT(m, i, j) ( (m).data[(i)*(m).stride + (j)] ) 

Matrix m_alloc(size_t rows, size_t cols); 
Matrix m_row_at(Matrix m, size_t);
void m_dot(Matrix result, Matrix x, Matrix y);
void m_sum(Matrix dst, Matrix src);
void m_copy(Matrix dst, Matrix src);
void m_fill(Matrix m, float val);
void m_fill_rand(Matrix m, float min, float max);
void m_sigf(Matrix m);
void m_print(Matrix m, const char* name);
#define M_PRINT(m) m_print(m, #m);

#ifdef MATRIX_IMPLEMENTATION_ 


#define M_ALLOC_ASSERT(rows, cols) do {\
	assert( (rows) != 0); \
	assert( (cols) != 0); \
} while(0)
Matrix m_alloc(size_t rows, size_t cols)
{
	M_ALLOC_ASSERT(rows, cols);

	Matrix m;
	m.num_rows = rows;
	m.num_cols = cols;
	m.stride = cols;
	m.data = malloc(sizeof(*m.data)* rows * cols);	
	assert(m.data);

	return m;
}




#define M_ROW_AT_ASSERT(m, row) do {\
	assert( ( (m).data != NULL) );\
	assert( (row) < m.num_rows);\
} while(0)
Matrix m_row_at(Matrix m, size_t row)
{
	M_ROW_AT_ASSERT(m, row);
	return (Matrix) {
		.num_rows = 1,
		.num_cols = m.num_cols,
		.stride = m.stride,
		.data = &M_ELEM_AT(m, row, 0),	
	};
}





#define DOT_ASSERT(result, x, y) do {\
	assert( (result).data != NULL ); \
	assert( (x).data      != NULL ); \
	assert( (y).data      != NULL ); \
	\
	assert( (x).num_cols      == (y).num_rows); \
	assert( (result).num_rows == (x).num_rows); \
	assert( (result).num_cols == (y).num_cols); \
} while(0)
void m_dot(Matrix result, Matrix x, Matrix y)
{
	DOT_ASSERT(result, x, y);

	//1 x (2<-----> 2) x 1 
	float inner_size = x.num_cols;
		
	for(size_t row = 0; row < result.num_rows; ++row)
       	{
		for(size_t col = 0; col < result.num_cols; ++col)
	       	{
			M_ELEM_AT(result, row, col) = 0; 
			for(size_t inner = 0; inner < inner_size; ++inner)	{
				M_ELEM_AT(result, row, col) += M_ELEM_AT(x, row, inner) * M_ELEM_AT(y, inner, col);
			}
		}
	}

	return;
}





#define SUM_ASSERT(dst, src) do {\
	assert( (dst).data != NULL); \
	assert( (src).data != NULL); \
	\
	assert( (dst).num_rows == (src).num_rows); \
	assert( (dst).num_cols == (src).num_cols); \
} while(0)
void m_sum(Matrix dst, Matrix src)
{
	SUM_ASSERT(dst, src);
	for(size_t row = 0; row < dst.num_rows; ++row)
		for(size_t col = 0; col < dst.num_cols; ++col)
			M_ELEM_AT(dst, row, col) += M_ELEM_AT(src, row, col); 
	
	return;	
}




#define COPY_ASSERT(dst, src) do {\
	assert( (dst).num_rows == (src).num_rows); \
	assert( (dst).num_cols == (src).num_cols); \
} while(0) 
void m_copy(Matrix dst, Matrix src)
{
	COPY_ASSERT(dst, src);

	for(size_t row = 0; row < dst.num_rows; ++row)
		for(size_t col = 0; col < dst.num_cols; ++col)	
			M_ELEM_AT(dst, row, col) = M_ELEM_AT(src, row, col);
	
	return;
}




#define FILL_ASSERT(m) do { \
	assert( (m).data != NULL); \
} while(0)
void m_fill(Matrix m, float val)
{
	FILL_ASSERT(m);

	for(size_t row = 0; row < m.num_rows; ++row)
		for(size_t col = 0; col < m.num_cols; ++col)
			M_ELEM_AT(m, row, col) = val; 
}




#define FILL_RAND_ASSERT(m, min, man) do { \
	assert( (m).data != NULL); \
	assert( (min) < (max) ); \
} while(0)
void m_fill_rand(Matrix m, float min, float max)
{
	FILL_RAND_ASSERT(m, min, max);

	for(size_t row = 0; row < m.num_rows; ++row)
		for(size_t col = 0; col < m.num_cols; ++col)
			M_ELEM_AT(m, row, col) = rand_float() * (max - min) + min; 
	
	return;
}




#define SIGF_ASSERT(m) do{\
	assert( (m).data != NULL);\
} while(0)
void m_sigf(Matrix m)
{
	for(size_t row = 0; row < m.num_rows; ++row)
		for(size_t col = 0; col < m.num_cols; ++col)
			M_ELEM_AT(m, row, col) = sigmoidf(M_ELEM_AT(m, row, col)); 	

	return;
}




#define PRINT_ASSERT(m) do {\
	assert( (m).data != NULL); \
	assert( (m).num_rows != 0);\
	assert( (m).num_cols != 0);\
} while(0)
void m_print(Matrix m, const char* name)
{
//	PRINT_ASSERT(m);

	printf("\t%s = {\n", name);
	for(size_t row = 0; row < m.num_rows; ++row)	
	{
		printf("\t\t");
		for(size_t col = 0; col < m.num_cols; ++col)	
			printf("%f  ", M_ELEM_AT(m, row, col));		
		printf("\n");
	}
	printf("\t}\n\n");

	return;
}
#endif //MATRIX_IMPLEMENTATION_
#endif //MATRIX_H_
