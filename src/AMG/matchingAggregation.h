#pragma once

namespace matchingAggregationContext{
    vector<itype> *M_buffer;
    vector<vtype> *ws_buffer;
    vector<itype> *mutex_buffer;

    void initContext(itype n){

      matchingAggregationContext::M_buffer = Vector::init<itype>(n , true, true);
      matchingAggregationContext::ws_buffer = Vector::init<vtype>(n , true, true);
      matchingAggregationContext::mutex_buffer = Vector::init<itype>(n , true, true);

    }

    __inline__
    void setBufferSize_matchingPairAggregation(itype n){

      matchingAggregationContext::M_buffer->n = n;
      matchingAggregationContext::ws_buffer->n = n;
      matchingAggregationContext::mutex_buffer->n = n;
    }

    void freeContext(){

      Vector::free(matchingAggregationContext::M_buffer);
      Vector::free(matchingAggregationContext::ws_buffer);
      Vector::free(matchingAggregationContext::mutex_buffer);

    }
}
