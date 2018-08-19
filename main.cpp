/**
 * This file is part of Gauss-Newton Solver.
 *
 * Copyright (C) 2018-2020 Dongsheng Yang <ydsf16@buaa.edu.cn> (Beihang University)
 * For more information see <https://github.com/ydsf16/Steepest-Descent-Gradient-Method>
 *
 * gradient_method is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * gradient_method is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gradient_method. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <chrono>

/* 计时类 */
class Runtimer{
public:
    inline void start()
    {
        t_s_  = std::chrono::steady_clock::now();
    }
    
    inline void stop()
    {
        t_e_ = std::chrono::steady_clock::now();
    }
    
    inline double duration()
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(t_e_ - t_s_).count() * 1000.0;
    }
    
private:
    std::chrono::steady_clock::time_point t_s_; //start time ponit
    std::chrono::steady_clock::time_point t_e_; //stop time point
};

/* 梯度下降类 */
class GradientMethod{
public:
    GradientMethod(double*a, double*b, double*c, double*d):
    a_(a), b_(b), c_(c), d_(d)
    {
        max_iter_= 1000;
        min_gradient_ = 1e-3;
        alpha_ = 1e-3;
    }
    
    void setParameters(int max_iter, double min_gradient, double alpha, bool is_out)
    {
        max_iter_ = max_iter;
        min_gradient_ = min_gradient;
        alpha_ = alpha;
        is_out_ = is_out;
    }
    
    void addObservation(const double& x, const double& y)
    {
        obs_.push_back( Eigen::Vector2d(x, y));
    }
    
    void calcGradient() //计算梯度
    {
        double g_a = 0.0, g_b = 0.0, g_c = 0.0, g_d = 0.0;
        
        /* 计算梯度 */
        for( size_t i = 0; i < obs_.size(); i ++)
        {
            Eigen::Vector2d& ob = obs_.at(i);
            const double& x= ob(0);
            const double& y = ob(1);
            
            g_a += - x*x*x * ( y - *a_*x*x*x - *b_*x*x - *c_*x - *d_ );
            g_b += - x*x * ( y - *a_*x*x*x - *b_*x*x - *c_*x - *d_ );
            g_c += - x*( y - *a_*x*x*x - *b_*x*x - *c_*x - *d_ );
            g_d += -  ( y - *a_*x*x*x - *b_*x*x - *c_*x - *d_ );
        }
        
        Gradient_(0) = g_a;
        Gradient_(1) = g_b;
        Gradient_(2) = g_c;
        Gradient_(3) = g_d;
    }
    double getCost()
    {
        double sum_cost = 0;
        for( size_t i = 0; i < obs_.size(); i ++)
        {
            Eigen::Vector2d& ob = obs_.at(i);
            const double& x= ob(0);
            const double& y = ob(1);
            double r = y - *a_*x*x*x - *b_*x*x - *c_*x - *d_;
            sum_cost += r*r;
        }
        return sum_cost;
    }
    void solve()
    {
        double sumt =0;
        bool is_conv = false;
        
        for( int i = 0; i < max_iter_; i ++)
        {
            Runtimer t;
            t.start();
            calcGradient();
            double dg = sqrt(Gradient_.transpose() * Gradient_);
            if( dg*alpha_  < min_gradient_ ) 
            {
                is_conv = true;
                break;
            }
            /* update */
            *a_ += -alpha_ * Gradient_(0);
            *b_ += -alpha_ * Gradient_(1);
            *c_ += -alpha_ * Gradient_(2);
            *d_ += -alpha_ * Gradient_(3);
                   
            t.stop();
            if( is_out_ )
            {
                std::cout << "Iter: " << std::left <<std::setw(3) << i << " Result: "<< std::left <<std::setw(10)  << *a_ << " " << std::left <<std::setw(10)  << *b_ << " " << std::left <<std::setw(10) << *c_ <<
                " " << std::left <<std::setw(10) << *d_ <<
                " Gradient: " << std::left <<std::setw(14) << dg << " cost: "<< std::left <<std::setw(14)  << getCost() << " time: " << std::left <<std::setw(14) << t.duration()  <<
                " total_time: "<< std::left <<std::setw(14) << (sumt += t.duration()) << std::endl;
            }
        }
        if( is_conv  == true)
            std::cout << "\nConverged\n\n";
        else
            std::cout << "\nDiverged\n\n";
    }
    
    
private:
    /* 要求解的四个参数 */
    double *a_, *b_, *c_, *d_;
    
    /* parameters */
    int max_iter_;
    double min_gradient_;
    double alpha_;
    
    /* 观测 */
    std::vector<Eigen::Vector2d> obs_;
    
    /* Gradient */
    Eigen::Vector4d Gradient_;
        
    /* 是否输出中间结果 */
    bool is_out_;
};//class GradientMethod


int main(int argc, char **argv) {
    const double aa = 4, bb = 3, cc = 2, dd = 1;// 实际方程的参数
    double a =0.0, b=0.0, c=0.0, d = 0.0; // 初值
    
    /* 构造问题 */
    GradientMethod gm(&a, &b, &c, &d);
    gm.setParameters(80000, 8e-5, 8e-7, true); //步长的选择非常重要，如果太大就会导致发散, 太小收敛速度极慢。
    
    /* 制造数据 */
    const size_t N = 1000; //数据个数
    cv::RNG rng(cv::getTickCount());
    for( size_t i = 0; i < N; i ++)
    {
        /* 生产带有高斯噪声的数据　*/
        double x = rng.uniform(0.0, 4.0) ;
        double y = aa*x*x*x + bb*x*x + cc*x + dd + rng.gaussian(1);
        
        /* 添加到观测中　*/
        gm.addObservation(x, y);
    }
    /* 用梯度下降法求解 */
    gm.solve();
    
    return 0;
}
