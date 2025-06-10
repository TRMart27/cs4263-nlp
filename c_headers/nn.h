#ifndef NN_H_
#define NN_H_

#include "util.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

typedef struct NN
{
	size_t count;
	Matrix* weight;
	Matrix* bias;   
	Matrix* active; //count +1 (to account for the final output)
} NN;

#define NN_INPUT(network) (network).active[0]
#define NN_OUTPUT(network) (network).active[(network).count]

NN nn_alloc(size_t* arch, size_t arch_len);
void nn_fill_rand(NN network, float min, float max);
void nn_forward(NN network);
float nn_loss(NN network, Matrix train, Matrix test);
void nn_error(NN network, NN gradient, float eps, Matrix train, Matrix test); 
void nn_learn(NN network, NN gradient, float learning_rate);
void nn_train(NN network, NN gradient, float eps, float learning_rate, float n_iter, Matrix train, Matrix test);
void nn_train_verbose(NN network, NN gradient, float eps, float learning_rate, float n_iter, Matrix train, Matrix test);
void gate_verify(NN network, float threshold);
void nn_print(NN network, const char* name);
#define NN_PRINT(network) nn_print(network, #network);


#ifdef NN_IMPLEMENTATION_



#define NN_ALLOC_ASSERT(network) do{\
	assert(network.weight != NULL);\
	assert(network.bias != NULL);\
	assert(network.active != NULL);\
	\
} while(0)
NN nn_alloc(size_t *arch, size_t arch_len)
{
 	assert(arch_len > 0);	
	NN network;
	network.count = arch_len - 1; 	//arch_len accounts for input layer
	
	network.weight  = malloc(sizeof(*network.weight) * network.count);
	network.bias    = malloc(sizeof(*network.bias)   * network.count);
	network.active  = malloc(sizeof(*network.active) * arch_len);
	NN_ALLOC_ASSERT(network);

	network.active[0] = m_alloc(1, arch[0]);
	for(size_t i = 1; i <= network.count; ++i)
	{
		network.weight[i - 1]  = m_alloc(network.active[i - 1].num_cols, arch[i]);
		network.bias[i - 1]    = m_alloc(1, arch[i]);
		network.active[i]  = m_alloc(1, arch[i]);
	}
	
	return network;
}



#define NN_FILL_RAND_ASSERT(network, min, max) do{\
	assert(min >= 0);\
	assert(max > min);\
} while(0)
void nn_fill_rand(NN network, float min, float max)
{
	NN_FILL_RAND_ASSERT(network, min, max);

	for(size_t i = 0; i < network.count; ++i)
	{ 
		m_fill_rand(network.weight[i], min, max);
		m_fill_rand(network.bias[i], min, max);
	}

	return;
}




void nn_forward(NN network)
{
	for(size_t i = 0; i < network.count; ++i)
	{
		m_dot(network.active[i + 1], network.active[i], network.weight[i]);
		m_sum(network.active[i + 1], network.bias[i]);
		m_sigf(network.active[i + 1]);
	}

	return;
}




#define NN_LOSS_ASSERT(network, train, test) do{\
	assert(train.data != NULL);\
	assert(test.data != NULL);\
} while(0)
float nn_loss(NN network, Matrix train, Matrix test)
{
	NN_LOSS_ASSERT(network, train, test);

	float result = 0.0f;
	float num_rows = train.num_rows;

	for(size_t row = 0; row < num_rows; ++row)
	{
		Matrix curr_row = m_row_at(train, row);
		m_copy(NN_INPUT(network), curr_row);

		nn_forward(network);


		Matrix expected_output = m_row_at(test, row);
		
		float prediction = M_ELEM_AT(NN_OUTPUT(network), 0, 0);
		float expected = M_ELEM_AT(expected_output, 0, 0);

		float error = prediction - expected;
		result += (error * error);
	}
	return (result) / num_rows;
}




#define NN_ERROR_ASSERT();
void nn_error(NN network, NN gradient, float eps, Matrix train, Matrix test) 
{
	float saved_val;
	float curr_loss = nn_loss(network, train, test);

	for(size_t i = 0; i < network.count; ++i) 
	{
		//for the weight matricies
		for(size_t row = 0; row < network.weight[i].num_rows; ++row)
		{		
			for(size_t col = 0; col < network.weight[i].num_cols; ++col)
			{
				saved_val = M_ELEM_AT(network.weight[i], row, col);
				M_ELEM_AT(network.weight[i], row, col) += eps;
				M_ELEM_AT(gradient.weight[i], row, col) = ( nn_loss(network, train, test) - curr_loss) / eps;
				M_ELEM_AT(network.weight[i], row, col) = saved_val;
			}
		}
		//bias matricies now
		for(size_t row = 0; row < network.bias[i].num_rows; ++row)
		{		
			for(size_t col = 0; col < network.bias[i].num_cols; ++col)
			{
				saved_val = M_ELEM_AT(network.bias[i], row, col);
				M_ELEM_AT(network.bias[i], row, col) += eps;
				M_ELEM_AT(gradient.bias[i], row, col) = ( nn_loss(network, train, test) - curr_loss) / eps;
				M_ELEM_AT(network.bias[i], row, col) = saved_val;
			}
		}
	}

	return;	
}




#define NN_LEARN_ASSERT(network, gradient, learning_rate) do{\
	assert(learning_rate != 0);\
	assert(network.count != 0);\
} while(0)
void nn_learn(NN network, NN gradient, float learning_rate)
{
	NN_LEARN_ASSERT(network, gradient, learning_rate);

	for(size_t i = 0; i < network.count; ++i) 	
	{
		for(size_t row = 0; row < network.weight[i].num_rows; ++row)
			for(size_t col = 0; col < network.weight[i].num_cols; ++col)	
				M_ELEM_AT(network.weight[i], row, col) -= learning_rate * M_ELEM_AT(gradient.weight[i], row, col);
		
	
		for(size_t row = 0; row < network.bias[i].num_rows; ++row)
			for(size_t col = 0; col < network.bias[i].num_cols; ++col)		
				M_ELEM_AT(network.bias[i], row, col) -= learning_rate * M_ELEM_AT(gradient.bias[i], row, col);
	}

	return;	
}



#define NN_TRAIN_ASSERT(network, gradient, eps, learning_rate, n_iter, train, test) do{\
	assert(network.count != 0);\
	assert(gradient.count != 0);\
	assert(eps != 0);\
	assert(learning_rate != 0);\
} while(0)
///TO DO: add file saving for the weights during each iteration, so that plotting the training loss is possible
void nn_train(NN network, NN gradient, float eps, float learning_rate, float n_iter, Matrix train, Matrix test)
{

	NN_TRAIN_ASSERT(network, gradient, eps, learning_rate, n_iter, train, test);
        
//	printf("\n----------------------------------------------------------------------------------------------------------\nTRAINING...\n %0.0f iterations\n\n", n_iter);
//	printf("\tInitial Loss    = { %.04f }\n\n", nn_loss(network, train, test));


        for(size_t i = 0; i < n_iter; ++i)
        {
                nn_error(network, gradient, eps, train, test);
                nn_learn(network, gradient, learning_rate);
        }
//        printf("\tFinal Loss    = { %.04f }\n\n", nn_loss(network, train, test));
//        printf("----------------------------------------------------------------------------------------------------------\n\n\n");

        return;
}



//uses the same training assertions as the non-verbose version of the function. There is literally a one line differce
///TO DO: add file saving for the weights during each iteration, so that plotting the training loss is possible
void nn_train_verbose(NN network, NN gradient, float eps, float learning_rate, float n_iter, Matrix train, Matrix test)
{
	NN_TRAIN_ASSERT(network, gradient, eps, learning_rate, n_iter, train, test);

        printf("\n----------------------------------------------------------------------------------------------------------\nTRAINING...\n %0.0f iterations\n\n", n_iter);
        printf("\tInitial Loss    = { %.04f }\n\n", nn_loss(network, train, test));


        for(size_t i = 0; i < n_iter; ++i)
        {
                nn_error(network, gradient, eps, train, test);
                nn_learn(network, gradient, learning_rate);
                printf("%zu ::  Loss = %f\n", i, nn_loss(network, train, test));
        }
        printf("\tFinal Loss    = { %.04f }\n\n", nn_loss(network, train, test));
        printf("----------------------------------------------------------------------------------------------------------\n\n\n");

        return;
}




void gate_verify(NN network, float threshold)
{
        printf("\n----------------------------------------------------------------------------------------------------------\nVERIFYING LOGIC GATE...\n\n");
        for(size_t i = 0; i < 2; ++i)
        {
                for(size_t j = 0; j < 2; ++j)
                {
                        M_ELEM_AT(NN_INPUT(network), 0, 0) = i;
                        M_ELEM_AT(NN_INPUT(network), 0, 1) = j;
                        nn_forward(network);
			
			M_ELEM_AT(NN_OUTPUT(network), 0, 0) = (M_ELEM_AT(NN_OUTPUT(network), 0, 0) > threshold) ? 1 : 0;

                        printf("%zu ^ %zu | %0.0f\n", i, j, M_ELEM_AT(NN_OUTPUT(network), 0, 0));
                }
        }
        printf("----------------------------------------------------------------------------------------------------------\n\n\n");

        return;
}



#define NN_PRINT_ASSERT(network, name) do{\
	assert( (network).weight != NULL);\
	assert( (network).bias != NULL)\
	assert( (network).active != NULL)\
} while(0)
void nn_print(NN network, const char* name)
{
	char buf[255];
	printf("%s {\n", name);
	for(size_t i = 0; i < network.count; ++i)
	{
		snprintf(buf, sizeof(buf), "weight %zu", i);
		m_print(network.weight[i], buf);
		snprintf(buf, sizeof(buf), "bias %zu", i);
		m_print(network.bias[i], buf);
	}
	printf("}\n");
}




#endif //NN_IMPLEMENTATION_
#endif //NN_H_
