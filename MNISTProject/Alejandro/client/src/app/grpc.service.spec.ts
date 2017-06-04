import { TestBed, inject } from '@angular/core/testing';
import { GrpcService } from './grpc.service';

describe('GrpcService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [GrpcService]
    });
  });

  it('should ...', inject([GrpcService], (service: GrpcService) => {
    expect(service).toBeTruthy();
  }));
});
