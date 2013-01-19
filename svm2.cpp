#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>
#include <malloc.h>
#include <string.h>
#include <locale.h>
#include <stdarg.h>
#include <float.h>
#include "svm.h"
int libsvm_version = LIBSVM_VERSION;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
void (*svm_print_string) (const char *) = &print_string_stdout;

void info(const char *fmt,...)
{
		char buf[BUFSIZ];
		va_list ap;
		va_start(ap,fmt);
		vsprintf(buf,fmt,ap);
		va_end(ap);
		(*svm_print_string)(buf);
}
void sigmoid_train( int l, const double *dec_values, const double *labels, double& A, double& B)
{
		double prior1=0, prior0 = 0;
		int i;

		for (i=0;i<l;i++)
			if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
		int max_iter=100;	// Maximal number of iterations
		double min_step=1e-10;	// Minimal step taken in line search
		double sigma=1e-12;	// For numerically strict PD of Hessian
		double eps=1e-5;
		double hiTarget=(prior1+1.0)/(prior1+2.0);
		double loTarget=1/(prior0+2.0);
		double *t=Malloc(double,l);
		double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
		double newA,newB,newf,d1,d2;
		int iter; 
	
		// Initial Point and Initial Fun Value
		A=0.0; B=log((prior0+1.0)/(prior1+1.0));
		double fval = 0.0;

		for (i=0;i<l;i++)
		{
			if (labels[i]>0) t[i]=hiTarget;
			else t[i]=loTarget;
			fApB = dec_values[i]*A+B;
			if (fApB>=0)
				fval += t[i]*fApB + log(1+exp(-fApB));
			else
				fval += (t[i] - 1)*fApB +log(1+exp(fApB));
		}
		for (iter=0;iter<max_iter;iter++)
		{
			// Update Gradient and Hessian (use H' = H + sigma I)
			h11=sigma; // numerically ensures strict PD
			h22=sigma;
			h21=0.0;g1=0.0;g2=0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*A+B;
				if (fApB >= 0)
				{
					p=exp(-fApB)/(1.0+exp(-fApB));
					q=1.0/(1.0+exp(-fApB));
				}
				else
				{
					p=1.0/(1.0+exp(fApB));
					q=exp(fApB)/(1.0+exp(fApB));
				}
				d2=p*q;
				h11+=dec_values[i]*dec_values[i]*d2;
				h22+=d2;
				h21+=dec_values[i]*d2;
				d1=t[i]-p;
				g1+=dec_values[i]*d1;
				g2+=d1;
			}

			// Stopping Criteria
			if (fabs(g1)<eps && fabs(g2)<eps)
				break;

			// Finding Newton direction: -inv(H') * g
			det=h11*h22-h21*h21;
			dA=-(h22*g1 - h21 * g2) / det;
			dB=-(-h21*g1+ h11 * g2) / det;
			gd=g1*dA+g2*dB;
	

			stepsize = 1;		// Line Search
			while (stepsize >= min_step)
			{
				newA = A + stepsize * dA;
				newB = B + stepsize * dB;

				// New function value
				newf = 0.0;
				for (i=0;i<l;i++)
				{
					fApB = dec_values[i]*newA+newB;
					if (fApB >= 0)
						newf += t[i]*fApB + log(1+exp(-fApB));
					else
						newf += (t[i] - 1)*fApB +log(1+exp(fApB));
				}
				// Check sufficient decrease
				if (newf<fval+0.0001*stepsize*gd)
				{
					A=newA;B=newB;fval=newf;
					break;
				}
				else
					stepsize = stepsize / 2.0;
			}		

	if (stepsize < min_step)
	{
		info("Line search fails in two-class probability estimates\n");
		break;
	}
}
double sigmoid_predict(double decision_value, double A, double B)
{
		double fApB = decision_value*A+B;
		// 1-p used later; avoid catastrophic cancellation
		if (fApB >= 0)
			return exp(-fApB)/(1.0+exp(-fApB));
		else
			return 1.0/(1+exp(fApB)) ;
}
static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};
double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}


int main()
{
	//struct svm_problem subprob;
	int iter = 0;
	int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	int counter = min(l,1000)+1;
	struct svm_node
	{
		int index;
		double value;
	};
	struct svm_parameter
	{
		int svm_type;
		int kernel_type;
		int degree;	/* for poly */
		double gamma;	/* for poly/rbf/sigmoid */
		double coef0;	/* for poly/sigmoid */

		/* these are for training only */
		double cache_size; /* in MB */
		double eps;	/* stopping criteria */
		double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
		int nr_weight;		/* for C_SVC */
		int *weight_label;	/* for C_SVC */
		double* weight;		/* for C_SVC */
		double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
		double p;	/* for EPSILON_SVR */
		int shrinking;	/* use the shrinking heuristics */
		int probability; /* do probability estimates */
	};
	//struct svm_model *submodel = svm_train(&subprob,param);
	

		if (iter>=max_iter)
			info("Reaching maximal iterations in two-class probability estimates\n");
		free(t);
	}

	
	int svm_save_model(const char *model_file_name, const svm_model *model)
	{
		FILE *fp = fopen("svmtrain.model","w");
		if(fp==NULL) return -1;

		char *old_locale = strdup(setlocale(LC_ALL, NULL));
		setlocale(LC_ALL, "C");

		const svm_parameter& param = model->param;

		fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
		fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

		if(param.kernel_type == POLY)
			fprintf(fp,"degree %d\n", param.degree);

		if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
			fprintf(fp,"gamma %g\n", param.gamma);

		if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
			fprintf(fp,"coef0 %g\n", param.coef0);

		int nr_class = model->nr_class;
		int l = model->l;
		fprintf(fp, "nr_class %d\n", nr_class);
		fprintf(fp, "total_sv %d\n",l);
		
		{
			fprintf(fp, "rho");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fprintf(fp," %g",model->rho[i]);
			fprintf(fp, "\n");
		}
		
		if(model->label)
		{
			fprintf(fp, "label");
			for(int i=0;i<nr_class;i++)
				fprintf(fp," %d",model->label[i]);
			fprintf(fp, "\n");
		}

		if(model->probA) // regression has probA only
		{
			fprintf(fp, "probA");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fprintf(fp," %g",model->probA[i]);
			fprintf(fp, "\n");
		}
		if(model->probB)
		{
			fprintf(fp, "probB");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fprintf(fp," %g",model->probB[i]);
			fprintf(fp, "\n");
		}

		if(model->nSV)
		{
			fprintf(fp, "nr_sv");
			for(int i=0;i<nr_class;i++)
				fprintf(fp," %d",model->nSV[i]);
			fprintf(fp, "\n");
		}

		fprintf(fp, "SV\n");
		const double * const *sv_coef = model->sv_coef;
		const svm_node * const *SV = model->SV;

		for(int i=0;i<l;i++)
		{
			for(int j=0;j<nr_class-1;j++)
				fprintf(fp, "%.16g ",sv_coef[j][i]);

			const svm_node *p = SV[i];

			if(param.kernel_type == PRECOMPUTED)
				fprintf(fp,"0:%d ",(int)(p->value));
			else
				while(p->index != -1)
				{
					fprintf(fp,"%d:%.8g ",p->index,p->value);
					p++;
				}
			fprintf(fp, "\n");
		}

		setlocale(LC_ALL, old_locale);
		free(old_locale);

		if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
		else return 0;
	}
	double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
	{
		int i;
		if(model->param.svm_type == ONE_CLASS || model->param.svm_type == EPSILON_SVR ||model->param.svm_type == NU_SVR)
		{
			double *sv_coef = model->sv_coef[0];
			double sum = 0;
			for(i=0;i<model->l;i++)
				sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
			sum -= model->rho[0];
			*dec_values = sum;

			if(model->param.svm_type == ONE_CLASS)
				return (sum>0)?1:-1;
			else
				return sum;
		}
		else
		{
			int nr_class = model->nr_class;
			int l = model->l;
			
			double *kvalue = Malloc(double,l);
			for(i=0;i<l;i++)
				kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

			int *start = Malloc(int,nr_class);
			start[0] = 0;
			for(i=1;i<nr_class;i++)
				start[i] = start[i-1]+model->nSV[i-1];

			int *vote = Malloc(int,nr_class);
			for(i=0;i<nr_class;i++)
				vote[i] = 0;

			int p=0;
			for(i=0;i<nr_class;i++)
				for(int j=i+1;j<nr_class;j++)
				{
					double sum = 0;
					int si = start[i];
					int sj = start[j];
					int ci = model->nSV[i];
					int cj = model->nSV[j];
					
					int k;
					double *coef1 = model->sv_coef[j-1];
					double *coef2 = model->sv_coef[i];
					for(k=0;k<ci;k++)
						sum += coef1[si+k] * kvalue[si+k];
					for(k=0;k<cj;k++)
						sum += coef2[sj+k] * kvalue[sj+k];
					sum -= model->rho[p];
					dec_values[p] = sum;

					if(dec_values[p] > 0)
						++vote[i];
					else
						++vote[j];
					p++;
				}

			int vote_max_idx = 0;
			for(i=1;i<nr_class;i++)
				if(vote[i] > vote[vote_max_idx])
					vote_max_idx = i;

			free(kvalue);
			free(start);
			free(vote);
			return model->label[vote_max_idx];
		}
	}
	getch();
	return 0;
}