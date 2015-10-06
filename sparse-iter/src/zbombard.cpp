/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt

       @precisions normal z -> s d c
*/

#include "common_magmasparse.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex general matrix A.
    This is a GPU implementation of the iterative bombardment suggested in
    Barrett et al.
    ''Algorithmic bombardment for the iterative solution of linear systems: 
      A poly-iterative approach''
    using BiCGSTAB, CGS and MQR. At this point, it only works for symmetric A.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in,out]
    x           magma_z_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgesv
    ********************************************************************/

extern "C" magma_int_t
magma_zbombard(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // 1=QMR, 2=CGS, 3+BiCGSTAB
    magma_int_t flag = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_BOMBARD;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    // solver variables
    double nom0, r0, res, Q_res, C_res, B_res, nomb;
    magmaDoubleComplex Q_rho = c_one, Q_rho1 = c_one, Q_eta = -c_one , Q_pds = c_one, 
                        Q_thet = c_one, Q_thet1 = c_one, Q_epsilon = c_one, 
                        Q_beta = c_one, Q_delta = c_one, Q_pde = c_one, Q_rde = c_one,
                        Q_gamm = c_one, Q_gamm1 = c_one, Q_psi = c_one;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // need to transpose the matrix
    
    // GPU workspace
    // QMR
    magma_z_matrix Q_r={Magma_CSR}, r_tld={Magma_CSR}, Q_x={Magma_CSR},
                    Q_v={Magma_CSR}, Q_w={Magma_CSR}, Q_wt={Magma_CSR},
                    Q_d={Magma_CSR}, Q_s={Magma_CSR}, Q_z={Magma_CSR}, Q_q={Magma_CSR}, 
                    Q_p={Magma_CSR}, Q_pt={Magma_CSR}, Q_y={Magma_CSR};
                    
    // CGS
    magma_z_matrix C_r={Magma_CSR}, C_rt={Magma_CSR},
                    C_p={Magma_CSR}, C_q={Magma_CSR}, C_u={Magma_CSR}, C_v={Magma_CSR},  C_t={Magma_CSR},
                    C_p_hat={Magma_CSR}, C_q_hat={Magma_CSR}, C_u_hat={Magma_CSR}, C_v_hat={Magma_CSR};

                    
                    
    CHECK( magma_zvinit( &r_tld, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    // QMR
    CHECK( magma_zvinit( &Q_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_w, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_wt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_pt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_x, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    // QMR
    CHECK( magma_zvinit( &C_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_rt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_x,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_p_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_q_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_u, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_u_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_v_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    
    
    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r_tld, &nom0, queue));
    solver_par->init_res = nom0;
    res = nom0;
    
    // QMR
    magma_zcopy( dofs, r_tld.dval, 1, Q_r.dval, 1 );   
    magma_zcopy( dofs, r_tld.dval, 1, Q_y.dval, 1 );   
    magma_zcopy( dofs, r_tld.dval, 1, Q_v.dval, 1 );  
    magma_zcopy( dofs, r_tld.dval, 1, Q_wt.dval, 1 );   
    magma_zcopy( dofs, r_tld.dval, 1, Q_z.dval, 1 ); 
    magma_zcopy( dofs, x->dval, 1, Q_x.dval, 1 ); 
    
    // CGS
    magma_zcopy( dofs, r_tld.dval, 1, C_r.dval, 1 );   
    magma_zcopy( dofs, x->dval, 1, C_x.dval, 1 ); 
    
    nomb = magma_dznrm2( dofs, b.dval, 1 );
    if ( nomb == 0.0 ){
        nomb=1.0;
    }       
    if ( (r0 = nomb * solver_par->rtol) < ATOLERANCE ){
        r0 = ATOLERANCE;
    }
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom0 < r0 ) {
        goto cleanup;
    }

    Q_psi = magma_zsqrt( magma_zdotc(dofs, Q_z.dval, 1, Q_z.dval, 1) );
    Q_rho = magma_zsqrt( magma_zdotc(dofs, Q_y.dval, 1, Q_y.dval, 1) );
    
        // v = y / rho
        // y = y / rho
        // w = wt / psi
        // z = z / psi
    magma_zqmr_1(  
    b.num_rows, 
    b.num_cols, 
    Q_rho,
    Q_psi,
    Q_y.dval, 
    Q_z.dval,
    Q_v.dval,
    Q_w.dval,
    queue );
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        
            //QMR: delta = z' * y;
        Q_delta = magma_zdotc(dofs, Q_z.dval, 1, Q_y.dval, 1);
        
            //CGS: rho = r' * r_tld
        C_rho = magma_zdotc(dofs, C_r.dval, 1, r_tld.dval, 1);

        
        if( solver_par->numiter == 1 ){
                //QMR: p = y;
                //QMR: q = z;
            magma_zcopy( dofs, Q_y.dval, 1, Q_p.dval, 1 );
            magma_zcopy( dofs, Q_z.dval, 1, Q_q.dval, 1 );
            
                //QMR: u = r;
                //QMR: p = r;
            magma_zcgs_2(  
            b.num_rows, 
            b.num_cols, 
            C_r.dval,
            C_u.dval,
            C_p.dval,
            queue );
        }
        else{
            Q_pde = Q_psi * Q_delta / Q_epsilon;
            Q_rde = Q_rho * MAGMA_Z_CNJG(Q_delta/Q_epsilon);
            
            C_beta = C_rho / C_rho_l;  
            
                //QMR p = y - pde * p
                //QMR q = z - rde * q
            magma_zqmr_2(  
            b.num_rows, 
            b.num_cols, 
            Q_pde,
            Q_rde,
            Q_y.dval,
            Q_z.dval,
            Q_p.dval, 
            Q_q.dval, 
            queue );
            
                  //CGS: u = r + beta*q;
                  //CGS: p = u + beta*( q + beta*p );
            magma_zcgs_1(  
            b.num_rows, 
            b.num_cols, 
            C_beta,
            C_r.dval,
            C_q.dval, 
            C_u.dval,
            C_p.dval,
            queue );
        }
        //QMR
        CHECK( magma_z_spmv( c_one, A, Q_p, c_zero, Q_pt, queue ));
        //CGS
        CHECK( magma_z_spmv( c_one, A, C_p, c_zero, C_v_hat, queue ));

        
            //QMR: epsilon = q' * pt;
        Q_epsilon = magma_zdotc(dofs, Q_q.dval, 1, Q_pt.dval, 1);
        Q_beta = Q_epsilon / Q_delta;
            //CGS: alpha = r_tld' * v_hat
        C_alpha = rho / magma_zdotc(dofs, r_tld.dval, 1, C_v_hat.dval, 1);
        
            //QMR: v = pt - beta * v
            //QMR: y = v
        magma_zqmr_3(  
        b.num_rows, 
        b.num_cols, 
        Q_beta,
        Q_pt.dval,
        Q_v.dval,
        Q_y.dval,
        queue );
        
            //CGS: q = u - alpha v_hat
            //CGS: t = u + q
        magma_zcgs_3(  
        b.num_rows, 
        b.num_cols, 
        C_alpha,
        C_v_hat.dval,
        C_u.dval, 
        C_q.dval,
        C_t.dval, 
        queue );
        
        Q_rho1 = Q_rho;      
            //QMR rho = norm(y);
        Q_rho = magma_zsqrt( magma_zdotc(dofs, Q_y.dval, 1, Q_y.dval, 1) );
        
            //QMR wt = A' * q - beta' * w;
        CHECK( magma_z_spmv( c_one, A, Q_q, c_zero, Q_wt, queue ));
            //CGS t = A u_hat
        CHECK( magma_z_spmv( c_one, A, C_t, c_zero, C_rt, queue ));   
        
        magma_zaxpy(dofs, - MAGMA_Z_CNJG( Q_beta ), Q_w.dval, 1, Q_wt.dval, 1);  
        
                    // no precond: z = wt
        magma_zcopy( dofs, Q_wt.dval, 1, Q_z.dval, 1 );
        


        Q_thet1 = Q_thet;        
        Q_thet = Q_rho / (Q_gamm * MAGMA_Z_MAKE( MAGMA_Z_ABS(Q_beta), 0.0 ));
        Q_gamm1 = Q_gamm;        
        
        Q_gamm = c_one / magma_zsqrt(c_one + Q_thet*Q_thet);        
        Q_eta = - Q_eta * Q_rho1 * Q_gamm * Q_gamm / (Q_beta * Q_gamm1 * Q_gamm1);        
        
        if( solver_par->numiter == 1 ){
            
                //QMR: d = eta * p + pds * d;
                //QMR: s = eta * pt + pds * d;
                //QMR: x = x + d;
                //QMR: r = r - s;
            magma_zqmr_4(  
            b.num_rows, 
            b.num_cols, 
            Q_eta,
            Q_p.dval,
            Q_pt.dval,
            Q_d.dval, 
            Q_s.dval, 
            Q_x.dval, 
            Q_r.dval, 
            queue );
        }
        else{

            Q_pds = (Q_thet1 * Q_gamm) * (Q_thet1 * Q_gamm);
            
                // d = eta * p + pds * d;
                // s = eta * pt + pds * d;
                // x = x + d;
                // r = r - s;
            magma_zqmr_5(  
            b.num_rows, 
            b.num_cols, 
            Q_eta,
            Q_pds,
            Q_p.dval,
            Q_pt.dval,
            Q_d.dval, 
            Q_s.dval, 
            Q_x.dval, 
            Q_r.dval, 
            queue );
        }
        
        
        // CGS: r = r -alpha*A u_hat
        // CGS: x = x + alpha u_hat
        magma_zcgs_4(  
        b.num_rows, 
        b.num_cols, 
        C_alpha,
        C_t.dval,
        C_rt.dval,
        C_x->dval, 
        C_r.dval,
        queue );
        
            //QMR: psi = norm(z);
        Q_psi = magma_zsqrt( magma_zdotc(dofs, Q_z.dval, 1, Q_z.dval, 1) );
        
            //QMR: v = y / rho
            //QMR: y = y / rho
            //QMR: w = wt / psi
            //QMR: z = z / psi
        magma_zqmr_1(  
        b.num_rows, 
        b.num_cols, 
        Q_rho,
        Q_psi,
        Q_y.dval, 
        Q_z.dval,
        Q_v.dval,
        Q_w.dval,
        queue );
        
        Q_res = magma_dznrm2( dofs, Q_r.dval, 1 );
        C_res = magma_dznrm2( dofs, C_r.dval, 1 );
        
        if( Q_res < res ){
            res = Q_res;
            flag = 1;
        }
        if( C_res < res ){
            res = C_res;
            flag = 2;
        }
       /* if( B_res < res ){
            res = B_res;
            flag = 3;
        }
    res = Q_res;*/
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( res/nomb <= solver_par->rtol || res <= solver_par->atol ){
            break;
        }
        
                

 
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    // copy back the best solver
    magma_zcopy( dofs, Q_x.dval, 1, x->dval, 1 ); 
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double Q_residual;
    CHECK(  magma_zresidualvec( A, b, *x, &Q_r, &Q_residual, queue));
    solver_par->iter_res = Q_res;
    solver_par->final_res = Q_residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->rtol*solver_par->init_res ||
            solver_par->iter_res < solver_par->atol ) {
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_zmfree(&r_tld, queue );
    // QMR
    magma_zmfree(&Q_r, queue );
    magma_zmfree(&Q_v,  queue );
    magma_zmfree(&Q_w,  queue );
    magma_zmfree(&Q_wt, queue );
    magma_zmfree(&Q_d,  queue );
    magma_zmfree(&Q_s,  queue );
    magma_zmfree(&Q_z,  queue );
    magma_zmfree(&Q_q,  queue );
    magma_zmfree(&Q_p,  queue );
    magma_zmfree(&Q_pt, queue );
    magma_zmfree(&Q_y,  queue );
    magma_zmfree(&Q_x,  queue );
    // CGS
    magma_zmfree(&C_r, queue );
    magma_zmfree(&C_rt, queue );
    magma_zmfree(&C_x, queue );
    magma_zmfree(&C_p, queue );
    magma_zmfree(&C_q, queue );
    magma_zmfree(&C_u, queue );
    magma_zmfree(&C_v, queue );
    magma_zmfree(&C_t, queue );
    magma_zmfree(&C_p_hat, queue );
    magma_zmfree(&C_q_hat, queue );
    magma_zmfree(&C_u_hat, queue );
    magma_zmfree(&C_v_hat, queue );

    
    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
}   /* magma_zbombard */